// SPDX-License-Identifier: Apache-2.0
//! Vulkan compute pipeline cache.
//!
//! A `PipelineCache` holds one `VkPipeline` per SPIR-V shader variant.
//! All pipelines share a single `VkDescriptorSetLayout` with 12 storage
//! buffer bindings (matching the llama.cpp convention).

use ash::vk;
use std::collections::HashMap;

/// Maximum number of storage buffer bindings per pipeline.
pub const MAX_BINDINGS: u32 = 12;

/// A compiled `VkPipeline` together with its layout and workgroup parameters.
pub struct Pipeline {
    pub name: String,
    pub module: vk::ShaderModule,
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    /// Number of storage buffer bindings this shader uses.
    pub binding_count: u32,
}

/// One allocation of push-constant data (up to 128 bytes).
pub type PushConstants = Vec<u8>;

/// Manages a set of compute pipelines for one logical device.
pub struct PipelineCache {
    device: ash::Device,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pipelines: HashMap<String, Pipeline>,
}

impl PipelineCache {
    /// Create a new pipeline cache for the given device.
    ///
    /// `shader_spvs`: map from shader name → SPIR-V bytes.
    pub fn new(
        device: ash::Device,
        shader_spvs: &HashMap<&str, &[u8]>,
    ) -> Result<Self, String> {
        // One global descriptor set layout with MAX_BINDINGS storage buffers.
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..MAX_BINDINGS)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let dsl_ci = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&bindings);
        let dsl = unsafe { device.create_descriptor_set_layout(&dsl_ci, None) }
            .map_err(|e| format!("create_descriptor_set_layout: {e}"))?;

        let mut cache = PipelineCache {
            device: device.clone(),
            descriptor_set_layout: dsl,
            pipelines: HashMap::new(),
        };

        // Compile all provided shaders.
        // RMSNorm shaders need do_multiply specialization variants.
        const RMSNORM_SHADERS: &[&str] = &[
            "rms_norm_f32",
        ];

        // Shaders using layout(local_size_x_id=0) need BLOCK_SIZE specialization.
        const MATVEC_SHADERS: &[&str] = &[
            "mul_mat_vec_f32_f32_f32",
            "mul_mat_vec_f32_f32_f32_subgroup",
            "mul_mat_vec_f16_f32_f32",
            "mul_mat_vec_f16_f32_f32_subgroup",
        ];

        let mut failed = 0usize;
        for (&name, &spv) in shader_spvs {
            let result = if RMSNORM_SHADERS.contains(&name) {
                // Compile both do_multiply=false and do_multiply=true variants.
                cache.compile_rms_norm(name, spv)
            } else if MATVEC_SHADERS.contains(&name) {
                cache.compile_matvec(name, spv)
            } else {
                cache.compile_one(name, spv)
            };
            if let Err(e) = result {
                log::warn!("Shader '{name}' failed to compile: {e}");
                failed += 1;
            }
        }
        // Account for the extra _mul variants created by compile_rms_norm.
        let total = shader_spvs.len() + RMSNORM_SHADERS.len();
        log::info!(
            "PipelineCache: {}/{} shaders compiled successfully.",
            total - failed,
            total
        );

        Ok(cache)
    }

    fn compile_one(&mut self, name: &str, spv: &[u8]) -> Result<(), String> {
        self.compile_one_with_spec(name, spv, &[])
    }

    fn compile_one_with_spec(
        &mut self,
        name: &str,
        spv: &[u8],
        spec_constants: &[(u32, u32)], // (constantID, value) pairs
    ) -> Result<(), String> {
        // Validate SPIR-V alignment.
        if spv.len() % 4 != 0 {
            return Err(format!("SPIR-V for '{name}' has unaligned size {}", spv.len()));
        }
        // Re-interpret as &[u32].
        let code: &[u32] = unsafe {
            std::slice::from_raw_parts(spv.as_ptr() as *const u32, spv.len() / 4)
        };

        let module_ci = vk::ShaderModuleCreateInfo::default().code(code);
        let module = unsafe { self.device.create_shader_module(&module_ci, None) }
            .map_err(|e| format!("create_shader_module for '{name}': {e}"))?;

        // Push constant range: up to 128 bytes (Vulkan guaranteed minimum).
        let pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(128);

        let layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&self.descriptor_set_layout))
            .push_constant_ranges(std::slice::from_ref(&pc_range));
        let layout = unsafe { self.device.create_pipeline_layout(&layout_ci, None) }
            .map_err(|e| format!("create_pipeline_layout for '{name}': {e}"))?;

        let entry = c"main";

        // Build specialization constant info if provided.
        // Each entry: {constantID, offset into data, size=4}
        // Data: tightly packed u32 values.
        // All allocations must outlive the vkCreateComputePipelines call.
        let spec_data: Vec<u32> = spec_constants.iter().map(|(_, v)| *v).collect();
        let spec_entries: Vec<vk::SpecializationMapEntry> = spec_constants
            .iter()
            .enumerate()
            .map(|(i, (id, _))| vk::SpecializationMapEntry {
                constant_id: *id,
                offset: (i * 4) as u32,
                size: 4,
            })
            .collect();

        let pipelines = if spec_constants.is_empty() {
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry);
            let pipeline_ci = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            unsafe {
                self.device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_ci),
                    None,
                )
            }
            .map_err(|(_, e)| format!("create_compute_pipelines for '{name}': {e}"))?
        } else {
            // spec_info borrows spec_entries and spec_data which are on this stack frame.
            let spec_info = vk::SpecializationInfo::default()
                .map_entries(&spec_entries)
                .data(bytemuck::cast_slice(&spec_data));
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(entry)
                .specialization_info(&spec_info);
            let pipeline_ci = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            unsafe {
                self.device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_ci),
                    None,
                )
            }
            .map_err(|(_, e)| format!("create_compute_pipelines for '{name}': {e}"))?
        };

        self.pipelines.insert(
            name.to_owned(),
            Pipeline {
                name: name.to_owned(),
                module,
                layout,
                pipeline: pipelines[0],
                binding_count: MAX_BINDINGS,
            },
        );
        Ok(())
    }

    /// Compile a shader with the standard matvec specialization constants:
    /// BLOCK_SIZE=512, NUM_ROWS=1, NUM_COLS=1.
    /// Also compiles a NUM_ROWS=4 variant for larger matrices.
    ///
    /// A NUM_ROWS=2 (`_r2`) variant used to be compiled here as well, but
    /// was never actually dispatched anywhere in this codebase (Rust or
    /// Python — every real matvec dispatch uses either this unsuffixed
    /// base variant, the `_subgroup` unsuffixed variant, or `_r4`; `grep
    /// -rn '_r2' src/ vllm_vulkan/ tests/` finds zero dispatch call
    /// sites, only its own compilation). Compiling it cost ~400ms
    /// (~22%) of `PipelineCache::new()`'s (i.e. model-load) wall time,
    /// measured directly (~1865-1885ms with it vs ~1461-1467ms without,
    /// consistently reproducible) before removing it — for a pipeline
    /// that was pure dead weight, so it's removed rather than kept "just
    /// in case". See `pipeline_cache_startup_tests::
    /// r2_matvec_variant_is_no_longer_compiled` for a (non-timing-based,
    /// CI-safe) regression guard confirming the removal.
    pub fn compile_matvec(&mut self, name: &str, spv: &[u8]) -> Result<(), String> {
        self.compile_one_with_spec(name, spv, &[
            (0, 512), // BLOCK_SIZE = 512
            (1, 1),   // NUM_ROWS = 1
            (2, 1),   // NUM_COLS = 1
        ])?;
        // NUM_ROWS=4 variant, BLOCK_SIZE=32 (not 512): BLOCK_SIZE is tied
        // directly to the shader's local_size_x (`layout(local_size_x_id =
        // 0, ...)`), so this is a genuinely independent tuning axis from
        // NUM_ROWS — measured across all 5 real Gemma4-E2B matvec shapes,
        // BLOCK_SIZE=32 is 1.05x-1.91x faster than the previous 512, and
        // consistently the best of {16, 32, 64, 128, 256, 512, 1024}
        // tried (16 regressed relative to 32 at every shape, consistent
        // with 32 being this hardware's native SIMD/subgroup width — see
        // matvec_r4_tests::r4_matches_base_at_gemma4_e2b_shapes /
        // r4_is_faster_than_base_at_gemma4_e2b_shapes, which validate
        // this exact configuration).
        // Unlike _r2 (compiled leniently above since nothing in production
        // actually dispatches it), _r4 is used for every single matvec
        // dispatch in the whole model — silently swallowing its
        // compilation failure here would defer the actual problem to a
        // much more confusing "shader not found" error the first time
        // something tries to dispatch it at runtime, instead of failing
        // fast and clearly at model-load time.
        let r4 = format!("{name}_r4");
        self.compile_one_with_spec(&r4, spv, &[(0, 32), (1, 4), (2, 1)])?;
        Ok(())
    }

    /// Compile rms_norm variants: plain (do_multiply=false) and weight-multiplying
    /// (do_multiply=true).  The weight-multiplying variant is registered as
    /// "<name>_mul".
    pub fn compile_rms_norm(&mut self, name: &str, spv: &[u8]) -> Result<(), String> {
        // Plain variant (do_multiply = false, the default)
        self.compile_one_with_spec(name, spv, &[])?;
        // Weight-multiplying variant (do_multiply = true, constant_id = 1)
        let mul_name = format!("{name}_mul");
        self.compile_one_with_spec(&mul_name, spv, &[
            (1, 1), // do_multiply = true
        ])
    }

    pub fn get(&self, name: &str) -> Option<&Pipeline> {
        self.pipelines.get(name)
    }

    pub fn pipeline_names(&self) -> Vec<String> {
        self.pipelines.keys().cloned().collect()
    }
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        unsafe {
            for p in self.pipelines.values() {
                self.device.destroy_pipeline(p.pipeline, None);
                self.device.destroy_pipeline_layout(p.layout, None);
                self.device.destroy_shader_module(p.module, None);
            }
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
