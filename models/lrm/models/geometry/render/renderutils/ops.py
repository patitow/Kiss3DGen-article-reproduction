# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import os
import sys
import torch
import torch.utils.cpp_extension

from .bsdf import *
from .loss import *

#----------------------------------------------------------------------------
# C++/Cuda plugin compiler/loader.

_cached_plugin = None
def _get_plugin():
    # Return cached plugin if already loaded.
    global _cached_plugin
    if _cached_plugin is not None:
        return _cached_plugin

    # Make sure we can find the necessary compiler and libary binaries.
    if os.name == 'nt':
        def find_cl_path():
            import glob
            # OBRIGATÓRIO: Priorizar VS 2019 (compatível com CUDA 12.1) - NÃO usar VS 2022
            search_paths = [
                # VS 2019 - PRIORIDADE MÁXIMA
                r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
                r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\*\bin\Hostx64\x64",
                r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\*\bin\Hostx64\x64",
                r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\*\bin\Hostx64\x64",
                r"C:\Program Files\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64",
                r"C:\Program Files\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\*\bin\Hostx64\x64",
                r"C:\Program Files\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\*\bin\Hostx64\x64",
                r"C:\Program Files\Microsoft Visual Studio\2019\Enterprise\VC\Tools\MSVC\*\bin\Hostx64\x64",
                # Fallback genérico VS 2019 (se caminho específico não funcionar)
                r"C:\Program Files (x86)\Microsoft Visual Studio\2019\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
                r"C:\Program Files\Microsoft Visual Studio\2019\*\VC\Tools\MSVC\*\bin\Hostx64\x64",
            ]
            for search_path in search_paths:
                paths = sorted(glob.glob(search_path), reverse=True)
                if paths:
                    print(f"[INFO] VS 2019 encontrado: {paths[0]}")
                    return paths[0]
            print("[ERRO] VS 2019 não encontrado! CUDA 12.1 requer VS 2019.")
            return None

        # If cl.exe is not on path, try to find it.
        # FORÇAR uso de VS 2019 para CUDA 12.1
            cl_path = find_cl_path()
            if cl_path is None:
            # Tentar verificar se cl.exe já está no PATH
            if os.system("where cl.exe >nul 2>nul") != 0:
                raise RuntimeError("VS 2019 não encontrado! CUDA 12.1 requer Visual Studio 2019. "
                                 "Instale VS 2019 Build Tools ou Community.")
            else:
                print("[AVISO] cl.exe encontrado no PATH, mas VS 2019 não detectado explicitamente")
        else:
            # Garantir que VS 2019 está no início do PATH
            current_path = os.environ.get('PATH', '')
            if cl_path not in current_path:
                os.environ['PATH'] = cl_path + ';' + current_path
                print(f"[INFO] VS 2019 adicionado ao INÍCIO do PATH: {cl_path}")
            else:
                # Mover para o início se já estiver
                path_parts = [p for p in current_path.split(';') if p]
                if cl_path in path_parts:
                    path_parts.remove(cl_path)
                os.environ['PATH'] = cl_path + ';' + ';'.join(path_parts)
                print(f"[INFO] VS 2019 movido para INÍCIO do PATH: {cl_path}")

    # Compiler options.
    opts = ['-DNVDR_TORCH']
    
    # CUDA-specific compiler options
    cuda_opts = ['-DNVDR_TORCH']
    
    # Windows-specific: Add flags to allow unsupported compiler (VS 2022 with CUDA 12.1)
    # CUDA 12.1 pode reclamar de VS 2022 mesmo sendo suportado - forçar flag
    if os.name == 'nt':
        # CRÍTICO: Adicionar flag ANTES de qualquer outra coisa para garantir que seja aplicado
        if '-allow-unsupported-compiler' not in cuda_opts:
            cuda_opts.insert(0, '-allow-unsupported-compiler')  # Inserir no início
        print("[INFO] Adicionando flag -allow-unsupported-compiler para compatibilidade VS 2022 + CUDA 12.1")

    # Linker options.
    if os.name == 'posix':
        ldflags = ['-lcuda', '-lnvrtc']
    elif os.name == 'nt':
        ldflags = ['cuda.lib', 'advapi32.lib', 'nvrtc.lib']

    # List of sources.
    source_files = [
        'c_src/mesh.cu',
        'c_src/loss.cu',
        'c_src/bsdf.cu',
        'c_src/normal.cu',
        'c_src/cubemap.cu',
        'c_src/common.cpp',
        'c_src/torch_bindings.cpp'
    ]

    # Some containers set this to contain old architectures that won't compile. We only need the one installed in the machine.
    # Limpar TORCH_CUDA_ARCH_LIST para permitir auto-detecção pelo PyTorch
    if 'TORCH_CUDA_ARCH_LIST' in os.environ:
        old_arch = os.environ['TORCH_CUDA_ARCH_LIST']
        if old_arch and old_arch.strip():
            print(f"[INFO] Limpando TORCH_CUDA_ARCH_LIST (era: '{old_arch}') para auto-detecção")
    os.environ['TORCH_CUDA_ARCH_LIST'] = ''
    
    # Garantir que CUDA_HOME está configurado corretamente
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"[INFO] renderutils_plugin: Usando CUDA_HOME={cuda_home}")
        # Verificar se nvcc existe
        nvcc_path = os.path.join(cuda_home, 'bin', 'nvcc.exe' if os.name == 'nt' else 'nvcc')
        if not os.path.exists(nvcc_path):
            print(f"[AVISO] nvcc não encontrado em {nvcc_path}")
    else:
        print("[AVISO] CUDA_HOME não configurado. PyTorch tentará detectar automaticamente.")

    # Try to detect if a stray lock file is left in cache directory and show a warning. This sometimes happens on Windows if the build is interrupted at just the right moment.
    try:
        lock_fn = os.path.join(torch.utils.cpp_extension._get_build_directory('renderutils_plugin', False), 'lock')
        if os.path.exists(lock_fn):
            print("Warning: Lock file exists in build directory: '%s'" % lock_fn)
    except:
        pass

    # Compile and load.
    source_paths = [os.path.join(os.path.dirname(__file__), fn) for fn in source_files]
    
    # Tentar compilar com tratamento de erro robusto
    try:
        # Garantir que flag está presente (já adicionado acima, mas verificar novamente)
        if '-allow-unsupported-compiler' not in cuda_opts:
            cuda_opts.insert(0, '-allow-unsupported-compiler')
            print("[INFO] Flag -allow-unsupported-compiler adicionado antes da compilação")
        
        # Para CUDA 12.x, adicionar flags adicionais
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH', '')
        if cuda_home and '12.' in cuda_home:
            print(f"[INFO] CUDA 12.x detectado ({cuda_home}), usando flags especiais para VS 2019")
            # Adicionar flags para suprimir warnings do compilador e garantir compatibilidade VS 2019
            if '-Xcompiler' not in ' '.join(cuda_opts):
                cuda_opts.extend(['-Xcompiler', '/wd4624', '-Xcompiler', '/wd4068', '-Xcompiler', '/wd4067'])
            # Garantir que flag de compatibilidade está presente
            if '-allow-unsupported-compiler' not in cuda_opts:
                cuda_opts.insert(0, '-allow-unsupported-compiler')
        elif cuda_home and '11.' in cuda_home:
            print(f"[INFO] CUDA 11.x detectado ({cuda_home}), usando flags para VS 2019")
            # CUDA 11.8 é compatível com VS 2019, mas pode precisar do flag
            if '-allow-unsupported-compiler' not in cuda_opts:
                cuda_opts.insert(0, '-allow-unsupported-compiler')
        
        print(f"[INFO] Compilando renderutils_plugin com flags: {cuda_opts[:5]}...")  # Mostrar primeiros flags
        
        # Capturar saída do ninja para diagnóstico
        import subprocess
        import sys
        
        try:
            # Configurar TORCH_CUDA_ARCH_LIST vazio para auto-detecção (nvdiffrast faz isso também)
            old_arch = os.environ.get('TORCH_CUDA_ARCH_LIST', '')
            os.environ['TORCH_CUDA_ARCH_LIST'] = ''  # PyTorch vai auto-detectar
            
            plugin = torch.utils.cpp_extension.load(name='renderutils_plugin', sources=source_paths, extra_cflags=opts,
         extra_cuda_cflags=cuda_opts, extra_ldflags=ldflags, with_cuda=True, verbose=True)
            
            # Restaurar TORCH_CUDA_ARCH_LIST se estava configurado
            if old_arch:
                os.environ['TORCH_CUDA_ARCH_LIST'] = old_arch
        except RuntimeError as e:
            error_msg = str(e)
            print(f"[ERRO] Erro durante compilação: {error_msg}")
            
            # Tentar capturar saída do ninja diretamente
            build_dir = torch.utils.cpp_extension._get_build_directory('renderutils_plugin', False)
            if os.path.exists(build_dir):
                print(f"[INFO] Verificando build directory: {build_dir}")
                # Procurar por arquivos de log ou saída do ninja
                for root, dirs, files in os.walk(build_dir):
                    for file in files:
                        if file.endswith('.log') or 'ninja' in file.lower() or file.endswith('.o') or file.endswith('.cu'):
                            log_path = os.path.join(root, file)
                            try:
                                if file.endswith('.log'):
                                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                                        log_content = f.read()
                                        if 'error' in log_content.lower() or 'fatal' in log_content.lower() or 'C1189' in log_content:
                                            print(f"[ERRO] Erro encontrado em {log_path}:")
                                            # Mostrar linhas com erros
                                            error_lines = [l for l in log_content.split('\n') if any(x in l.lower() for x in ['error', 'fatal', 'c1189', 'unsupported'])]
                                            for line in error_lines[-15:]:
                                                print(f"  {line}")
                            except Exception as log_err:
                                pass
            
            raise

    # Import, cache, and return the compiled module.
    import renderutils_plugin
    _cached_plugin = renderutils_plugin
    return _cached_plugin
    except Exception as e:
        error_msg = str(e)
        print(f"[ERRO] Falha ao compilar renderutils_plugin: {error_msg}")
        
        # Se o erro for relacionado a compilação CUDA, tentar limpar cache e recompilar
        if 'Error building extension' in error_msg or 'ninja' in error_msg.lower():
            print("[INFO] Tentando limpar cache de compilação e recompilar...")
            try:
                build_dir = torch.utils.cpp_extension._get_build_directory('renderutils_plugin', False)
                if os.path.exists(build_dir):
                    import shutil
                    print(f"[INFO] Removendo diretório de build: {build_dir}")
                    shutil.rmtree(build_dir, ignore_errors=True)
                
                # Tentar novamente com flags mais agressivos
                if '-allow-unsupported-compiler' not in cuda_opts:
                    cuda_opts.append('-allow-unsupported-compiler')
                
                torch.utils.cpp_extension.load(name='renderutils_plugin', sources=source_paths, extra_cflags=opts,
                     extra_cuda_cflags=cuda_opts, extra_ldflags=ldflags, with_cuda=True, verbose=True)
                
                import renderutils_plugin
                _cached_plugin = renderutils_plugin
                return _cached_plugin
            except Exception as e2:
                print(f"[ERRO] Recompilação também falhou: {e2}")
                print("[AVISO] renderutils_plugin não disponível. Pipeline usará fallback CPU.")
                raise RuntimeError(f"Não foi possível compilar renderutils_plugin. Erro: {e2}. "
                                 f"Pipeline requer esta extensão para renderização GPU. "
                                 f"Verifique se Visual Studio Build Tools está instalado corretamente.")
        
        raise

#----------------------------------------------------------------------------
# Internal kernels, just used for testing functionality

class _fresnel_shlick_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f0, f90, cosTheta):
        out = _get_plugin().fresnel_shlick_fwd(f0, f90, cosTheta, False)
        ctx.save_for_backward(f0, f90, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        f0, f90, cosTheta = ctx.saved_variables
        return _get_plugin().fresnel_shlick_bwd(f0, f90, cosTheta, dout) + (None,)

def _fresnel_shlick(f0, f90, cosTheta, use_python=False):
    if use_python:
        out = bsdf_fresnel_shlick(f0, f90, cosTheta)
    else:
        out = _fresnel_shlick_func.apply(f0, f90, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _fresnel_shlick contains inf or NaN"
    return out


class _ndf_ggx_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosTheta):
        out = _get_plugin().ndf_ggx_fwd(alphaSqr, cosTheta, False)
        ctx.save_for_backward(alphaSqr, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosTheta = ctx.saved_variables
        return _get_plugin().ndf_ggx_bwd(alphaSqr, cosTheta, dout) + (None,)

def _ndf_ggx(alphaSqr, cosTheta, use_python=False):
    if use_python:
        out = bsdf_ndf_ggx(alphaSqr, cosTheta)
    else:
        out = _ndf_ggx_func.apply(alphaSqr, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _ndf_ggx contains inf or NaN"
    return out

class _lambda_ggx_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosTheta):
        out = _get_plugin().lambda_ggx_fwd(alphaSqr, cosTheta, False)
        ctx.save_for_backward(alphaSqr, cosTheta)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosTheta = ctx.saved_variables
        return _get_plugin().lambda_ggx_bwd(alphaSqr, cosTheta, dout) + (None,)

def _lambda_ggx(alphaSqr, cosTheta, use_python=False):
    if use_python:
        out = bsdf_lambda_ggx(alphaSqr, cosTheta)
    else:
        out = _lambda_ggx_func.apply(alphaSqr, cosTheta)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _lambda_ggx contains inf or NaN"
    return out

class _masking_smith_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alphaSqr, cosThetaI, cosThetaO):
        ctx.save_for_backward(alphaSqr, cosThetaI, cosThetaO)
        out = _get_plugin().masking_smith_fwd(alphaSqr, cosThetaI, cosThetaO, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        alphaSqr, cosThetaI, cosThetaO = ctx.saved_variables
        return _get_plugin().masking_smith_bwd(alphaSqr, cosThetaI, cosThetaO, dout) + (None,)

def _masking_smith(alphaSqr, cosThetaI, cosThetaO, use_python=False):
    if use_python:
        out = bsdf_masking_smith_ggx_correlated(alphaSqr, cosThetaI, cosThetaO)
    else:
        out = _masking_smith_func.apply(alphaSqr, cosThetaI, cosThetaO)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of _masking_smith contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Shading normal setup (bump mapping + bent normals)

class _prepare_shading_normal_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl):
        ctx.two_sided_shading, ctx.opengl = two_sided_shading, opengl
        out = _get_plugin().prepare_shading_normal_fwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl, False)
        ctx.save_for_backward(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm)
        return out

    @staticmethod
    def backward(ctx, dout):
        pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm = ctx.saved_variables
        return _get_plugin().prepare_shading_normal_bwd(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, dout, ctx.two_sided_shading, ctx.opengl) + (None, None, None)

def prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading=True, opengl=True, use_python=False):
    '''Takes care of all corner cases and produces a final normal used for shading:
        - Constructs tangent space
        - Flips normal direction based on geometric normal for two sided Shading
        - Perturbs shading normal by normal map
        - Bends backfacing normals towards the camera to avoid shading artifacts

        All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        pos: World space g-buffer position.
        view_pos: Camera position in world space (typically using broadcasting).
        perturbed_nrm: Trangent-space normal perturbation from normal map lookup.
        smooth_nrm: Interpolated vertex normals.
        smooth_tng: Interpolated vertex tangents.
        geom_nrm: Geometric (face) normals.
        two_sided_shading: Use one/two sided shading
        opengl: Use OpenGL/DirectX normal map conventions 
        use_python: Use PyTorch implementation (for validation)
    Returns:
        Final shading normal
    '''    

    if perturbed_nrm is None:
        perturbed_nrm = torch.tensor([0, 0, 1], dtype=torch.float32, device='cuda', requires_grad=False)[None, None, None, ...]
    
    if use_python:
        out = bsdf_prepare_shading_normal(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    else:
        out = _prepare_shading_normal_func.apply(pos, view_pos, perturbed_nrm, smooth_nrm, smooth_tng, geom_nrm, two_sided_shading, opengl)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of prepare_shading_normal contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# BSDF functions

class _lambert_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nrm, wi):
        out = _get_plugin().lambert_fwd(nrm, wi, False)
        ctx.save_for_backward(nrm, wi)
        return out

    @staticmethod
    def backward(ctx, dout):
        nrm, wi = ctx.saved_variables
        return _get_plugin().lambert_bwd(nrm, wi, dout) + (None,)

def lambert(nrm, wi, use_python=False):
    '''Lambertian bsdf. 
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        nrm: World space shading normal.
        wi: World space light vector.
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded diffuse value with shape [minibatch_size, height, width, 1]
    '''

    if use_python:
        out = bsdf_lambert(nrm, wi)
    else:
        out = _lambert_func.apply(nrm, wi)
 
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of lambert contains inf or NaN"
    return out

class _frostbite_diffuse_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nrm, wi, wo, linearRoughness):
        out = _get_plugin().frostbite_fwd(nrm, wi, wo, linearRoughness, False)
        ctx.save_for_backward(nrm, wi, wo, linearRoughness)
        return out

    @staticmethod
    def backward(ctx, dout):
        nrm, wi, wo, linearRoughness = ctx.saved_variables
        return _get_plugin().frostbite_bwd(nrm, wi, wo, linearRoughness, dout) + (None,)

def frostbite_diffuse(nrm, wi, wo, linearRoughness, use_python=False):
    '''Frostbite, normalized Disney Diffuse bsdf. 
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent.

    Args:
        nrm: World space shading normal.
        wi: World space light vector.
        wo: World space camera vector.
        linearRoughness: Material roughness
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded diffuse value with shape [minibatch_size, height, width, 1]
    '''

    if use_python:
        out = bsdf_frostbite(nrm, wi, wo, linearRoughness)
    else:
        out = _frostbite_diffuse_func.apply(nrm, wi, wo, linearRoughness)
 
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of lambert contains inf or NaN"
    return out

class _pbr_specular_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, col, nrm, wo, wi, alpha, min_roughness):
        ctx.save_for_backward(col, nrm, wo, wi, alpha)
        ctx.min_roughness = min_roughness
        out = _get_plugin().pbr_specular_fwd(col, nrm, wo, wi, alpha, min_roughness, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        col, nrm, wo, wi, alpha = ctx.saved_variables
        return _get_plugin().pbr_specular_bwd(col, nrm, wo, wi, alpha, ctx.min_roughness, dout) + (None, None)

def pbr_specular(col, nrm, wo, wi, alpha, min_roughness=0.08, use_python=False):
    '''Physically-based specular bsdf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        col: Specular lobe color
        nrm: World space shading normal.
        wo: World space camera vector.
        wi: World space light vector
        alpha: Specular roughness parameter with shape [minibatch_size, height, width, 1]
        min_roughness: Scalar roughness clamping threshold

        use_python: Use PyTorch implementation (for validation)
    Returns:
        Shaded specular color
    '''

    if use_python:
        out = bsdf_pbr_specular(col, nrm, wo, wi, alpha, min_roughness=min_roughness)
    else:
        out = _pbr_specular_func.apply(col, nrm, wo, wi, alpha, min_roughness)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_specular contains inf or NaN"
    return out

class _pbr_bsdf_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF):
        ctx.save_for_backward(kd, arm, pos, nrm, view_pos, light_pos)
        ctx.min_roughness = min_roughness
        ctx.BSDF = BSDF
        out = _get_plugin().pbr_bsdf_fwd(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        kd, arm, pos, nrm, view_pos, light_pos = ctx.saved_variables
        return _get_plugin().pbr_bsdf_bwd(kd, arm, pos, nrm, view_pos, light_pos, ctx.min_roughness, ctx.BSDF, dout) + (None, None, None)

def pbr_bsdf(kd, arm, pos, nrm, view_pos, light_pos, min_roughness=0.08, bsdf="lambert", use_python=False):
    '''Physically-based bsdf, both diffuse & specular lobes
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        kd: Diffuse albedo.
        arm: Specular parameters (attenuation, linear roughness, metalness).
        pos: World space position.
        nrm: World space shading normal.
        view_pos: Camera position in world space, typically using broadcasting.
        light_pos: Light position in world space, typically using broadcasting.
        min_roughness: Scalar roughness clamping threshold
        bsdf: Controls diffuse BSDF, can be either 'lambert' or 'frostbite'

        use_python: Use PyTorch implementation (for validation)

    Returns:
        Shaded color.
    '''    

    BSDF = 0 
    if bsdf == 'frostbite':
        BSDF = 1

    if use_python:
        out = bsdf_pbr(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF)
    else:
        out = _pbr_bsdf_func.apply(kd, arm, pos, nrm, view_pos, light_pos, min_roughness, BSDF)
    
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of pbr_bsdf contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# cubemap filter with filtering across edges

class _diffuse_cubemap_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        out = _get_plugin().diffuse_cubemap_fwd(cubemap)
        ctx.save_for_backward(cubemap)
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, = ctx.saved_variables
        cubemap_grad = _get_plugin().diffuse_cubemap_bwd(cubemap, dout)
        return cubemap_grad, None

def diffuse_cubemap(cubemap, use_python=False):
    if use_python:
        assert False
    else:
        out = _diffuse_cubemap_func.apply(cubemap)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of diffuse_cubemap contains inf or NaN"
    return out

class _specular_cubemap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap, roughness, costheta_cutoff, bounds):
        out = _get_plugin().specular_cubemap_fwd(cubemap, bounds, roughness, costheta_cutoff)
        ctx.save_for_backward(cubemap, bounds)
        ctx.roughness, ctx.theta_cutoff = roughness, costheta_cutoff
        return out

    @staticmethod
    def backward(ctx, dout):
        cubemap, bounds = ctx.saved_variables
        cubemap_grad = _get_plugin().specular_cubemap_bwd(cubemap, bounds, dout, ctx.roughness, ctx.theta_cutoff)
        return cubemap_grad, None, None, None

# Compute the bounds of the GGX NDF lobe to retain "cutoff" percent of the energy
def __ndfBounds(res, roughness, cutoff):
    def ndfGGX(alphaSqr, costheta):
        costheta = np.clip(costheta, 0.0, 1.0)
        d = (costheta * alphaSqr - costheta) * costheta + 1.0
        return alphaSqr / (d * d * np.pi)

    # Sample out cutoff angle
    nSamples = 1000000
    costheta = np.cos(np.linspace(0, np.pi/2.0, nSamples))
    D = np.cumsum(ndfGGX(roughness**4, costheta))
    idx = np.argmax(D >= D[..., -1] * cutoff)

    # Brute force compute lookup table with bounds
    bounds = _get_plugin().specular_bounds(res, costheta[idx])

    return costheta[idx], bounds
__ndfBoundsDict = {}

def specular_cubemap(cubemap, roughness, cutoff=0.99, use_python=False):
    assert cubemap.shape[0] == 6 and cubemap.shape[1] == cubemap.shape[2], "Bad shape for cubemap tensor: %s" % str(cubemap.shape)

    if use_python:
        assert False
    else:
        key = (cubemap.shape[1], roughness, cutoff)
        if key not in __ndfBoundsDict:
            __ndfBoundsDict[key] = __ndfBounds(*key)
        out = _specular_cubemap.apply(cubemap, roughness, *__ndfBoundsDict[key])
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of specular_cubemap contains inf or NaN"
    return out[..., 0:3] / out[..., 3:]

#----------------------------------------------------------------------------
# Fast image loss function

class _image_loss_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, target, loss, tonemapper):
        ctx.loss, ctx.tonemapper = loss, tonemapper
        ctx.save_for_backward(img, target)
        out = _get_plugin().image_loss_fwd(img, target, loss, tonemapper, False)
        return out

    @staticmethod
    def backward(ctx, dout):
        img, target = ctx.saved_variables
        return _get_plugin().image_loss_bwd(img, target, dout, ctx.loss, ctx.tonemapper) + (None, None, None)

def image_loss(img, target, loss='l1', tonemapper='none', use_python=False):
    '''Compute HDR image loss. Combines tonemapping and loss into a single kernel for better perf.
    All tensors assume a shape of [minibatch_size, height, width, 3] or broadcastable equivalent unless otherwise noted.

    Args:
        img: Input image.
        target: Target (reference) image. 
        loss: Type of loss. Valid options are ['l1', 'mse', 'smape', 'relmse']
        tonemapper: Tonemapping operations. Valid options are ['none', 'log_srgb']
        use_python: Use PyTorch implementation (for validation)

    Returns:
        Image space loss (scalar value).
    '''
    if use_python:
        out = image_loss_fn(img, target, loss, tonemapper)
    else:
        out = _image_loss_func.apply(img, target, loss, tonemapper)
        out = torch.sum(out) / (img.shape[0]*img.shape[1]*img.shape[2])

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of image_loss contains inf or NaN"
    return out

#----------------------------------------------------------------------------
# Transform points function

class _xfm_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, matrix, isPoints):
        ctx.save_for_backward(points, matrix)
        ctx.isPoints = isPoints
        return _get_plugin().xfm_fwd(points, matrix, isPoints, False)

    @staticmethod
    def backward(ctx, dout):
        points, matrix = ctx.saved_variables
        return (_get_plugin().xfm_bwd(points, matrix, dout, ctx.isPoints),) + (None, None, None)

def xfm_points(points, matrix, use_python=False):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    
    if use_python:
        out = torch.matmul(torch.nn.functional.pad(points, pad=(0,1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    else:
        out = _xfm_func.apply(points, matrix, True)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out

def xfm_vectors(vectors, matrix, use_python=False):
    '''Transform vectors.
    Args:
        vectors: Tensor containing 3D vectors with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)

    Returns:
        Transformed vectors in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''    

    if use_python:
        out = torch.matmul(torch.nn.functional.pad(vectors, pad=(0,1), mode='constant', value=0.0), torch.transpose(matrix, 1, 2))[..., 0:3].contiguous()
    else:
        out = _xfm_func.apply(vectors, matrix, False)

    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_vectors contains inf or NaN"
    return out



