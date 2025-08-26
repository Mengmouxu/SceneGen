import importlib

__attributes = {
    'BasicTrainer': 'basic',
    
    'SparseStructureVaeTrainer': 'vae.sparse_structure_vae',
    
    'SLatVaeGaussianTrainer': 'vae.structured_latent_vae_gaussian',
    'SLatVaeRadianceFieldDecoderTrainer': 'vae.structured_latent_vae_rf_dec',
    'SLatVaeMeshDecoderTrainer': 'vae.structured_latent_vae_mesh_dec',
    
    'FlowMatchingTrainer': 'flow_matching.flow_matching',
    'FlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    'TextConditionedFlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    'ImageConditionedFlowMatchingCFGTrainer': 'flow_matching.flow_matching',
    'ImageConditionedVGGTFlowMatchingCFGTrainer': 'flow_matching.flow_matching',

    
    'SparseFlowMatchingTrainer': 'flow_matching.sparse_flow_matching',
    'SparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',
    'TextConditionedSparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',
    'ImageConditionedSparseFlowMatchingCFGTrainer': 'flow_matching.sparse_flow_matching',
}

__submodules = []

__all__ = list(__attributes.keys()) + __submodules

def __getattr__(name):
    if name not in globals():
        if name in __attributes:
            module_name = __attributes[name]
            module = importlib.import_module(f".{module_name}", __name__)
            globals()[name] = getattr(module, name)
        elif name in __submodules:
            module = importlib.import_module(f".{name}", __name__)
            globals()[name] = module
        else:
            raise AttributeError(f"module {__name__} has no attribute {name}")
    return globals()[name]


# For Pylance
if __name__ == '__main__':
    from .basic import BasicTrainer

    from .vae.sparse_structure_vae import SparseStructureVaeTrainer

    from .vae.structured_latent_vae_gaussian import SLatVaeGaussianTrainer
    from .vae.structured_latent_vae_rf_dec import SLatVaeRadianceFieldDecoderTrainer
    from .vae.structured_latent_vae_mesh_dec import SLatVaeMeshDecoderTrainer
    
    from .flow_matching.flow_matching import (
        FlowMatchingTrainer,
        FlowMatchingCFGTrainer,
        TextConditionedFlowMatchingCFGTrainer,
        ImageConditionedFlowMatchingCFGTrainer,
        ImageConditionedVGGTFlowMatchingCFGTrainer,
    )
    
    from .flow_matching.sparse_flow_matching import (
        SparseFlowMatchingTrainer,
        SparseFlowMatchingCFGTrainer,
        TextConditionedSparseFlowMatchingCFGTrainer,
        ImageConditionedSparseFlowMatchingCFGTrainer,
    )
