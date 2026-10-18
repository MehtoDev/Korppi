//! Pipeline state serializes with the `serde` feature (FIX-1.5; finding RHI-14), so materials
//! and pipeline presets can be stored in scenes and assets. Debug labels are not serialized.

use korppi_rhi::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BlendComponent, BlendFactor,
    BlendOp, BlendState, BorderColor, BufferUsage, ColorTargetState, ColorWriteMask,
    CompareFunction, CullMode, DepthStencilState, FilterMode, FrontFace, MultisampleState,
    PolygonMode, PrimitiveTopology, PushConstantRange, RasterizerState, SampleCount,
    SamplerAddressMode, SamplerDescriptor, ShaderStage, ShaderStages, StencilFaceState,
    StencilOperation, StorageTextureAccess, TextureFormat, TextureUsage, VertexAttribute,
    VertexBinding, VertexFormat, VertexLayoutDescriptor, VertexStepMode,
};
use serde::{de::DeserializeOwned, Serialize};
use std::fmt::Debug;

fn round_trip<T: Serialize + DeserializeOwned + PartialEq + Debug>(value: &T) {
    let json = serde_json::to_string(value).expect("serialize");
    let back: T = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(&back, value, "{json}");
}

#[test]
fn pipeline_state_round_trips() {
    round_trip(&ShaderStage::Fragment);
    round_trip(&(ShaderStages::VERTEX | ShaderStages::FRAGMENT));
    round_trip(&(BufferUsage::VERTEX | BufferUsage::COPY_DST));
    round_trip(&(TextureUsage::SAMPLED | TextureUsage::COLOR_ATTACHMENT));
    round_trip(&VertexLayoutDescriptor {
        bindings: vec![VertexBinding {
            binding: 0,
            stride: 20,
            step_mode: VertexStepMode::Instance,
        }],
        attributes: vec![VertexAttribute {
            location: 1,
            binding: 0,
            format: VertexFormat::Float32x3,
            offset: 8,
        }],
    });
    round_trip(&PrimitiveTopology::TriangleStrip);
    round_trip(&RasterizerState {
        cull_mode: CullMode::Front,
        front_face: FrontFace::Clockwise,
        polygon_mode: PolygonMode::Line,
        depth_bias: 1.5,
        slope_scaled_depth_bias: 2.0,
        ..RasterizerState::default()
    });
    round_trip(&DepthStencilState {
        format: TextureFormat::Depth24PlusStencil8,
        depth_compare: CompareFunction::GreaterEqual,
        stencil_front: StencilFaceState {
            compare: CompareFunction::Equal,
            fail_op: StencilOperation::Zero,
            depth_fail_op: StencilOperation::IncrementWrap,
            pass_op: StencilOperation::Replace,
        },
        stencil_read_mask: 0x0F,
        ..DepthStencilState::default()
    });
    round_trip(&ColorTargetState {
        format: TextureFormat::Rgba16Float,
        blend: Some(BlendState {
            color: BlendComponent {
                src_factor: BlendFactor::SrcAlpha,
                dst_factor: BlendFactor::OneMinusSrcAlpha,
                operation: BlendOp::Add,
            },
            alpha: BlendComponent {
                src_factor: BlendFactor::One,
                dst_factor: BlendFactor::Constant,
                operation: BlendOp::Max,
            },
        }),
        write_mask: ColorWriteMask::RED | ColorWriteMask::ALPHA,
    });
    round_trip(&MultisampleState {
        count: SampleCount::Count4,
        alpha_to_coverage: true,
    });
    round_trip(&PushConstantRange {
        stages: ShaderStages::COMPUTE,
        offset: 16,
        size: 64,
    });
    round_trip(&BindGroupLayoutEntry {
        binding: 3,
        ty: BindingType::StorageTexture {
            format: TextureFormat::R32Float,
            access: StorageTextureAccess::ReadWrite,
        },
        stages: ShaderStages::COMPUTE,
    });
}

#[test]
fn labels_are_not_serialized() {
    let sampler = SamplerDescriptor {
        min_filter: FilterMode::Linear,
        mag_filter: FilterMode::Nearest,
        mipmap_filter: FilterMode::Linear,
        address_mode_u: SamplerAddressMode::Repeat,
        address_mode_v: SamplerAddressMode::MirroredRepeat,
        address_mode_w: SamplerAddressMode::ClampToBorder,
        compare: Some(CompareFunction::Less),
        max_anisotropy: 8,
        border_color: BorderColor::OpaqueWhite,
        lod_min_clamp: 0.5,
        lod_max_clamp: 4.0,
        mip_lod_bias: -0.25,
        label: Some("shadow sampler"),
    };
    let json = serde_json::to_string(&sampler).expect("serialize");
    assert!(!json.contains("shadow sampler"), "{json}");
    let back: SamplerDescriptor = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        back,
        SamplerDescriptor {
            label: None,
            ..sampler
        }
    );

    let layout = BindGroupLayoutDescriptor {
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            ty: BindingType::UniformBuffer {
                dynamic_offset: true,
            },
            stages: ShaderStages::VERTEX,
        }],
        label: Some("camera"),
    };
    let json = serde_json::to_string(&layout).expect("serialize");
    let back: BindGroupLayoutDescriptor = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(
        back,
        BindGroupLayoutDescriptor {
            label: None,
            ..layout
        }
    );
}
