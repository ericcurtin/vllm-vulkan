// SPDX-License-Identifier: Apache-2.0
//! Vulkan device enumeration and management.
//!
//! On macOS, Vulkan calls are translated to Metal by KosmicKrisp (Mesa/Zink).
//! On Linux, native Vulkan is used directly.

use std::ffi::{c_char, CStr};
use std::mem;
use std::sync::OnceLock;

// OnceLock is used for the global Vulkan instance singleton (SAFE_INSTANCE).

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

// ────────────────────────────────────────────────────────────────────────────
// Vulkan FFI (only the handful of symbols we need)
// ────────────────────────────────────────────────────────────────────────────

type VkInstance = *mut std::ffi::c_void;
type VkPhysicalDevice = *mut std::ffi::c_void;
type VkResult = i32;

const VK_SUCCESS: i32 = 0;
const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_API_VERSION_1_2: u32 = (1 << 22) | (2 << 12);

#[repr(C)]
struct VkApplicationInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    p_application_name: *const c_char,
    application_version: u32,
    p_engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct VkInstanceCreateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    flags: u32,
    p_application_info: *const VkApplicationInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const c_char,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const c_char,
}

#[repr(C)]
struct VkPhysicalDeviceProperties {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: u32,
    device_name: [u8; 256],
    pipeline_cache_uuid: [u8; 16],
    limits: [u8; 504], // VkPhysicalDeviceLimits is large; approximate
    sparse_properties: [u8; 20],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VkPhysicalDeviceMemoryProperties {
    memory_type_count: u32,
    memory_types: [u8; 256],
    memory_heap_count: u32,
    memory_heaps: [[u8; 16]; 16], // VkMemoryHeap: size(u64) + flags(u32) + pad(u32)
}

// On macOS, KosmicKrisp ships libvulkan.dylib to /usr/local/lib.
// On Linux the standard Vulkan loader is libvulkan.so.
#[link(name = "vulkan")]
extern "C" {
    fn vkCreateInstance(
        create_info: *const VkInstanceCreateInfo,
        allocator: *const std::ffi::c_void,
        instance: *mut VkInstance,
    ) -> VkResult;

    fn vkDestroyInstance(instance: VkInstance, allocator: *const std::ffi::c_void);

    fn vkEnumeratePhysicalDevices(
        instance: VkInstance,
        count: *mut u32,
        devices: *mut VkPhysicalDevice,
    ) -> VkResult;

    fn vkGetPhysicalDeviceProperties(
        device: VkPhysicalDevice,
        properties: *mut VkPhysicalDeviceProperties,
    );

    fn vkGetPhysicalDeviceMemoryProperties(
        device: VkPhysicalDevice,
        properties: *mut VkPhysicalDeviceMemoryProperties,
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Global Vulkan instance (created once, never destroyed)
// ────────────────────────────────────────────────────────────────────────────

// Safety: VkInstance is an opaque pointer; we only create it once and share
// it read-only across threads after that.
unsafe impl Send for SafeInstance {}
unsafe impl Sync for SafeInstance {}

struct SafeInstance(VkInstance);

static SAFE_INSTANCE: OnceLock<Option<SafeInstance>> = OnceLock::new();

fn get_instance() -> Option<VkInstance> {
    let holder = SAFE_INSTANCE.get_or_init(|| unsafe { create_instance() });
    holder.as_ref().map(|si| si.0)
}

unsafe fn create_instance() -> Option<SafeInstance> {
    let app_name = b"vllm-vulkan\0";
    let engine_name = b"vllm-vulkan-rs\0";

    let app_info = VkApplicationInfo {
        s_type: VK_STRUCTURE_TYPE_APPLICATION_INFO,
        p_next: std::ptr::null(),
        p_application_name: app_name.as_ptr() as *const c_char,
        application_version: 1,
        p_engine_name: engine_name.as_ptr() as *const c_char,
        engine_version: 1,
        api_version: VK_API_VERSION_1_2,
    };

    let create_info = VkInstanceCreateInfo {
        s_type: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        p_next: std::ptr::null(),
        flags: 0,
        p_application_info: &app_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: std::ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: std::ptr::null(),
    };

    let mut instance: VkInstance = std::ptr::null_mut();
    let result = vkCreateInstance(&create_info, std::ptr::null(), &mut instance);
    if result == VK_SUCCESS && !instance.is_null() {
        Some(SafeInstance(instance))
    } else {
        log::warn!("vkCreateInstance failed (result={}); Vulkan unavailable", result);
        None
    }
}

fn enumerate_physical_devices_raw() -> Vec<VkPhysicalDevice> {
    let Some(instance) = get_instance() else {
        return vec![];
    };
    unsafe {
        let mut count: u32 = 0;
        if vkEnumeratePhysicalDevices(instance, &mut count, std::ptr::null_mut()) != VK_SUCCESS
            || count == 0
        {
            return vec![];
        }
        let mut devices = vec![std::ptr::null_mut::<std::ffi::c_void>(); count as usize];
        if vkEnumeratePhysicalDevices(instance, &mut count, devices.as_mut_ptr()) != VK_SUCCESS {
            return vec![];
        }
        devices.truncate(count as usize);
        devices
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Public device info types
// ────────────────────────────────────────────────────────────────────────────

/// Parsed information about a single Vulkan physical device.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub name: String,
    pub vendor_id: u32,
    pub device_type: String,
    pub api_version: String,
    pub driver_version: u32,
    pub total_memory_bytes: u64,
}

fn device_type_str(ty: u32) -> &'static str {
    match ty {
        0 => "other",
        1 => "integrated_gpu",
        2 => "discrete_gpu",
        3 => "virtual_gpu",
        4 => "cpu",
        _ => "unknown",
    }
}

fn api_version_str(v: u32) -> String {
    format!("{}.{}.{}", (v >> 22) & 0x7f, (v >> 12) & 0x3ff, v & 0xfff)
}

fn total_memory(mem_props: &VkPhysicalDeviceMemoryProperties) -> u64 {
    // Each heap is { size: u64, flags: u32, _pad: u32 } = 16 bytes.
    // We sum heaps that have the DEVICE_LOCAL flag (bit 0).
    let count = mem_props.memory_heap_count.min(16) as usize;
    let mut total: u64 = 0;
    for i in 0..count {
        let heap = &mem_props.memory_heaps[i];
        let size = u64::from_le_bytes(heap[0..8].try_into().unwrap_or([0; 8]));
        let flags = u32::from_le_bytes(heap[8..12].try_into().unwrap_or([0; 4]));
        if flags & 1 != 0 {
            total += size;
        }
    }
    // Unified-memory devices (e.g. Apple Silicon via KosmicKrisp) report heap
    // without DEVICE_LOCAL. Fall back to the largest heap in that case.
    if total == 0 {
        for i in 0..count {
            let heap = &mem_props.memory_heaps[i];
            let size = u64::from_le_bytes(heap[0..8].try_into().unwrap_or([0; 8]));
            total = total.max(size);
        }
    }
    total
}

pub fn enumerate_devices() -> Vec<DeviceInfo> {
    enumerate_physical_devices_raw()
        .into_iter()
        .map(|phys| unsafe {
            let mut props: VkPhysicalDeviceProperties = mem::zeroed();
            vkGetPhysicalDeviceProperties(phys, &mut props);

            let mut mem_props: VkPhysicalDeviceMemoryProperties = mem::zeroed();
            vkGetPhysicalDeviceMemoryProperties(phys, &mut mem_props);

            // Device name is a null-terminated UTF-8 string.
            let name = CStr::from_bytes_until_nul(&props.device_name)
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "Unknown".to_string());

            DeviceInfo {
                name,
                vendor_id: props.vendor_id,
                device_type: device_type_str(props.device_type).to_string(),
                api_version: api_version_str(props.api_version),
                driver_version: props.driver_version,
                total_memory_bytes: total_memory(&mem_props),
            }
        })
        .collect()
}

pub fn is_vulkan_available() -> bool {
    !enumerate_physical_devices_raw().is_empty()
}

pub fn device_count() -> usize {
    enumerate_physical_devices_raw().len()
}

pub fn synchronize_all() -> Result<(), String> {
    // Vulkan has no global sync; the Python layer synchronises per-queue.
    Ok(())
}

pub fn memory_info(device_idx: usize) -> Result<(u64, u64), String> {
    let devs = enumerate_devices();
    let info = devs
        .get(device_idx)
        .ok_or_else(|| format!("device index {} out of range", device_idx))?;
    // Without a VkDevice we cannot query actual used bytes; return (0, total).
    Ok((0, info.total_memory_bytes))
}

// ────────────────────────────────────────────────────────────────────────────
// PyO3 class
// ────────────────────────────────────────────────────────────────────────────

/// A Vulkan physical device handle exposed to Python.
#[pyclass]
pub struct VulkanDevice {
    #[pyo3(get)]
    pub index: usize,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub device_type: String,
    #[pyo3(get)]
    pub api_version: String,
    #[pyo3(get)]
    pub total_memory_bytes: u64,
}

#[pymethods]
impl VulkanDevice {
    #[new]
    pub fn new(index: usize) -> PyResult<Self> {
        let devs = enumerate_devices();
        let info = devs
            .get(index)
            .ok_or_else(|| PyRuntimeError::new_err(format!("no Vulkan device at index {index}")))?;
        Ok(Self {
            index,
            name: info.name.clone(),
            device_type: info.device_type.clone(),
            api_version: info.api_version.clone(),
            total_memory_bytes: info.total_memory_bytes,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "VulkanDevice(index={}, name={:?}, type={}, vk={})",
            self.index, self.name, self.device_type, self.api_version
        )
    }
}
