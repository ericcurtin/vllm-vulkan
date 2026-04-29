// SPDX-License-Identifier: Apache-2.0
//! Vulkan device enumeration and management using the `ash` crate.
//!
//! On macOS, Vulkan calls are translated to Metal by KosmicKrisp (Mesa/Zink).
//! On Linux, native Vulkan is used directly.
//!
//! This module:
//!  - Initialises a `VkInstance` (once, lazily) via `ash::Entry::load()`
//!  - Enumerates `VkPhysicalDevice`s and queries their properties
//!  - Creates `VkDevice` + `VkQueue`s on demand
//!  - Provides the `DeviceInfo` struct exposed to Python

use std::ffi::{CStr, CString, c_char};
use std::sync::OnceLock;

use ash::vk;
use log::{debug, warn};

// ─── Lazy global Vulkan instance ─────────────────────────────────────────────

struct VkState {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical_devices: Vec<vk::PhysicalDevice>,
}

// Safety: we hold no non-Send data; the raw pointers inside ash types are
// safe to share across threads as long as we don't mutate them concurrently
// (which our read-only enumeration does not do).
unsafe impl Send for VkState {}
unsafe impl Sync for VkState {}

static VK_STATE: OnceLock<Option<VkState>> = OnceLock::new();

fn get_state() -> Option<&'static VkState> {
    VK_STATE
        .get_or_init(|| unsafe { init_vulkan() })
        .as_ref()
}

unsafe fn init_vulkan() -> Option<VkState> {
    // Load the Vulkan loader library (libvulkan.so / MoltenVK.dylib).
    let entry = match ash::Entry::load() {
        Ok(e) => e,
        Err(e) => {
            debug!("Cannot load Vulkan loader: {e}");
            return None;
        }
    };

    // Application info (version 1.2 minimum).
    let app_name = CString::new("vllm-vulkan").unwrap();
    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_2);

    // Enumerate available instance extensions.
    let available_exts = entry
        .enumerate_instance_extension_properties(None)
        .unwrap_or_default();
    let available_ext_names: Vec<&CStr> = available_exts
        .iter()
        .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
        .collect();

    let mut enabled_exts: Vec<*const c_char> = Vec::new();
    let portability_ext = c"VK_KHR_portability_enumeration";
    let debug_utils_ext = ash::ext::debug_utils::NAME;

    let mut has_portability = false;
    if available_ext_names.iter().any(|n| *n == portability_ext) {
        enabled_exts.push(portability_ext.as_ptr());
        has_portability = true;
    }
    // Debug utils for validation layers (optional).
    if available_ext_names.iter().any(|n| *n == debug_utils_ext) {
        enabled_exts.push(debug_utils_ext.as_ptr());
    }

    let mut flags = vk::InstanceCreateFlags::empty();
    if has_portability {
        flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
    }

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&enabled_exts)
        .flags(flags);

    let instance = match entry.create_instance(&create_info, None) {
        Ok(i) => i,
        Err(e) => {
            debug!("vkCreateInstance failed: {e}");
            return None;
        }
    };

    let physical_devices = instance
        .enumerate_physical_devices()
        .unwrap_or_default();

    if physical_devices.is_empty() {
        debug!("No Vulkan physical devices found.");
        // Don't destroy instance; it's cheap and we may re-enumerate later.
        return None;
    }

    debug!(
        "Vulkan instance created. {} physical device(s) found.",
        physical_devices.len()
    );

    Some(VkState {
        _entry: entry,
        instance,
        physical_devices,
    })
}

// ─── Public DeviceInfo ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_type: String,
    pub api_version: String,
    pub driver_version: u32,
    pub total_memory_bytes: u64,
}

fn device_type_str(t: vk::PhysicalDeviceType) -> &'static str {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => "discrete",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "integrated",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "virtual",
        vk::PhysicalDeviceType::CPU => "cpu",
        _ => "other",
    }
}

fn api_version_str(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

fn total_device_memory(instance: &ash::Instance, pd: vk::PhysicalDevice) -> u64 {
    let mem_props = unsafe { instance.get_physical_device_memory_properties(pd) };
    let mut total = 0u64;
    for i in 0..mem_props.memory_heap_count as usize {
        let heap = &mem_props.memory_heaps[i];
        if heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL) {
            total += heap.size;
        }
    }
    // If nothing flagged as device-local (UMA), fall back to largest heap.
    if total == 0 {
        total = (0..mem_props.memory_heap_count as usize)
            .map(|i| mem_props.memory_heaps[i].size)
            .max()
            .unwrap_or(0);
    }
    total
}

// ─── Public API ──────────────────────────────────────────────────────────────

pub fn is_vulkan_available() -> bool {
    get_state().is_some()
}

pub fn device_count() -> usize {
    get_state()
        .map(|s| s.physical_devices.len())
        .unwrap_or(0)
}

pub fn enumerate_devices() -> Vec<DeviceInfo> {
    let Some(state) = get_state() else {
        return Vec::new();
    };
    state
        .physical_devices
        .iter()
        .map(|&pd| {
            let props = unsafe { state.instance.get_physical_device_properties(pd) };
            let name = unsafe {
                CStr::from_ptr(props.device_name.as_ptr())
                    .to_string_lossy()
                    .into_owned()
            };
            DeviceInfo {
                name,
                vendor_id: props.vendor_id,
                device_type: device_type_str(props.device_type).to_owned(),
                api_version: api_version_str(props.api_version),
                driver_version: props.driver_version,
                total_memory_bytes: total_device_memory(&state.instance, pd),
            }
        })
        .collect()
}

pub fn memory_info(device_idx: usize) -> Result<(u64, u64), String> {
    let state = get_state().ok_or_else(|| "Vulkan not available".to_owned())?;
    let &pd = state
        .physical_devices
        .get(device_idx)
        .ok_or_else(|| format!("no device at index {device_idx}"))?;

    let total = total_device_memory(&state.instance, pd);
    // Without creating a VkDevice + VK_EXT_memory_budget, "used" is unknown.
    Ok((0, total))
}

pub fn synchronize_all() -> Result<(), String> {
    // Without an active VkDevice/VkQueue, there is nothing to synchronise.
    // A future implementation will call vkDeviceWaitIdle on each open device.
    Ok(())
}

// ─── Logical device access (for compute) ─────────────────────────────────────

/// A Vulkan logical device with a compute queue.
pub struct ComputeDevice {
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub compute_queue: vk::Queue,
    pub compute_queue_family: u32,
    // Capabilities
    pub fp16: bool,
    pub subgroup_size: u32,
    pub max_workgroup_invocations: u32,
}

// ComputeDevice does NOT implement Drop — ownership of `device` is transferred
// to ComputeEngine, which handles cleanup via its own Drop impl.

impl ComputeDevice {
    /// Create a logical device for the physical device at `idx`.
    pub fn create(idx: usize) -> Result<Self, String> {
        let state = get_state().ok_or("Vulkan not available")?;
        let &pd = state
            .physical_devices
            .get(idx)
            .ok_or_else(|| format!("no device at index {idx}"))?;

        let instance = &state.instance;

        // Find a compute queue family.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(pd) };
        let compute_family = queue_families
            .iter()
            .enumerate()
            .find(|(_, qf)| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .map(|(i, _)| i as u32)
            .ok_or("No compute queue family found")?;

        let priorities = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_family)
            .queue_priorities(&priorities);

        // Query device capabilities.
        let props = unsafe { instance.get_physical_device_properties(pd) };

        // Enable fp16 if the device supports it.
        // We query the features into local structs, then extract raw copies.
        let (fp16, has_float64) = unsafe {
            let mut features16 = vk::PhysicalDevice16BitStorageFeatures::default();
            let p16 = &mut features16 as *mut vk::PhysicalDevice16BitStorageFeatures;
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut *p16);
            instance.get_physical_device_features2(pd, &mut features2);
            // Read results from the raw pointers — safe because Vulkan has
            // filled the structs and we are still within the unsafe block.
            let fp16_val = (*p16).storage_buffer16_bit_access == vk::TRUE;
            let f64_val  = features2.features.shader_float64 == vk::TRUE;
            (fp16_val, f64_val)
        };
        let fp16 = fp16 || has_float64; // rough heuristic

        let device_features = vk::PhysicalDeviceFeatures {
            shader_float64: if has_float64 { vk::TRUE } else { vk::FALSE },
            shader_int64: vk::TRUE,
            shader_int16: vk::TRUE,
            ..Default::default()
        };

        // Build extension list.
        let available = unsafe {
            instance
                .enumerate_device_extension_properties(pd)
                .unwrap_or_default()
        };
        let available_names: Vec<&CStr> = available
            .iter()
            .map(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) })
            .collect();

        let ext_16bit = c"VK_KHR_16bit_storage";
        let ext_storage8 = c"VK_KHR_8bit_storage";
        let ext_portability = c"VK_KHR_portability_subset";

        let mut exts: Vec<*const c_char> = Vec::new();
        if available_names.contains(&&*ext_16bit)   { exts.push(ext_16bit.as_ptr()); }
        if available_names.contains(&&*ext_storage8) { exts.push(ext_storage8.as_ptr()); }
        if available_names.contains(&&*ext_portability) { exts.push(ext_portability.as_ptr()); }

        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_ci))
            .enabled_extension_names(&exts)
            .enabled_features(&device_features);

        let device = unsafe { instance.create_device(pd, &device_ci, None) }
            .map_err(|e| format!("vkCreateDevice: {e}"))?;

        let compute_queue = unsafe { device.get_device_queue(compute_family, 0) };

        let subgroup_props = {
            let mut sp = vk::PhysicalDeviceSubgroupProperties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sp);
            unsafe { instance.get_physical_device_properties2(pd, &mut p2) };
            sp
        };

        Ok(ComputeDevice {
            instance: instance.clone(),
            physical_device: pd,
            device,
            compute_queue,
            compute_queue_family: compute_family,
            fp16,
            subgroup_size: subgroup_props.subgroup_size,
            max_workgroup_invocations: props.limits.max_compute_work_group_invocations,
        })
    }
}

// ─── Python-exposed VulkanDevice ─────────────────────────────────────────────

use pyo3::prelude::*;

#[pyclass]
pub struct VulkanDevice {
    pub index: usize,
    info: DeviceInfo,
}

#[pymethods]
impl VulkanDevice {
    #[getter]
    fn index(&self) -> usize { self.index }
    #[getter]
    fn name(&self) -> &str { &self.info.name }
    #[getter]
    fn device_type(&self) -> &str { &self.info.device_type }
    #[getter]
    fn api_version(&self) -> &str { &self.info.api_version }
    #[getter]
    fn total_memory_bytes(&self) -> u64 { self.info.total_memory_bytes }

    fn __repr__(&self) -> String {
        format!(
            "VulkanDevice(index={}, name={:?}, type={}, memory={:.1}GB)",
            self.index,
            self.info.name,
            self.info.device_type,
            self.info.total_memory_bytes as f64 / 1e9,
        )
    }
}

impl VulkanDevice {
    pub fn new(index: usize) -> Option<Self> {
        enumerate_devices()
            .into_iter()
            .nth(index)
            .map(|info| VulkanDevice { index, info })
    }
}
