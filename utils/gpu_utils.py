"""
GPU 자동 선택 유틸리티
사용 가능한 GPU를 자동으로 찾아서 사용
"""
import torch
import os
from typing import Optional, List

# NVML 오류 방지: 환경변수 설정
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')
# NVML 초기화 문제 회피
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')

def get_available_gpu(min_memory_mb: int = 1000, exclude_gpus: Optional[list] = None) -> int:
    """
    사용 가능한 GPU를 자동으로 찾아서 반환
    
    Args:
        min_memory_mb: 최소 필요한 메모리 (MB)
        exclude_gpus: 제외할 GPU 번호 리스트
    
    Returns:
        사용 가능한 GPU 번호 (없으면 -1)
    """
    if not torch.cuda.is_available():
        return -1
    
    exclude_gpus = exclude_gpus or []
    num_gpus = torch.cuda.device_count()
    
    # 환경변수에서 제외할 GPU 확인
    exclude_env = os.environ.get('EXCLUDE_GPUS', '')
    if exclude_env:
        exclude_gpus.extend([int(x.strip()) for x in exclude_env.split(',') if x.strip().isdigit()])
    
    best_gpu = -1
    max_free_memory = 0
    
    for gpu_id in range(num_gpus):
        if gpu_id in exclude_gpus:
            continue
        
        try:
            torch.cuda.set_device(gpu_id)
            # 메모리 사용량 확인
            memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2  # MB
            
            # GPU 메모리 정보 가져오기 (가능한 경우)
            try:
                props = torch.cuda.get_device_properties(gpu_id)
                total_memory = props.total_memory / 1024**2  # MB
                free_memory = total_memory - memory_reserved
            except:
                # 메모리 정보를 가져올 수 없으면 스킵
                continue
            
            if free_memory >= min_memory_mb and free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu = gpu_id
        except Exception as e:
            continue
    
    # 사용 가능한 GPU가 없으면 가장 사용량이 적은 GPU 선택
    if best_gpu == -1:
        for gpu_id in range(num_gpus):
            if gpu_id in exclude_gpus:
                continue
            try:
                torch.cuda.set_device(gpu_id)
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
                if best_gpu == -1 or memory_reserved < max_free_memory:
                    max_free_memory = memory_reserved
                    best_gpu = gpu_id
            except:
                continue
    
    return best_gpu if best_gpu != -1 else 0

def get_available_gpus(min_memory_mb: int = 2000, exclude_gpus: Optional[list] = None) -> List[int]:
    """사용 가능한 모든 GPU 리스트 반환 (nvidia-smi로 실제 메모리 사용량 확인)"""
    if not torch.cuda.is_available():
        return []
    
    exclude_gpus = exclude_gpus or []
    exclude_env = os.environ.get('EXCLUDE_GPUS', '')
    if exclude_env:
        exclude_gpus.extend([int(x.strip()) for x in exclude_env.split(',') if x.strip().isdigit()])
    
    # nvidia-smi로 실제 메모리 사용량 확인
    gpu_memory_info = {}
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        gpu_id = int(parts[0].strip())
                        memory_used = int(parts[1].strip())
                        memory_total = int(parts[2].strip())
                        gpu_memory_info[gpu_id] = {
                            'used': memory_used,
                            'total': memory_total,
                            'free': memory_total - memory_used
                        }
    except Exception:
        # nvidia-smi 실패 시 fallback
        pass
    
    available = []
    for gpu_id in range(torch.cuda.device_count()):
        if gpu_id in exclude_gpus:
            continue
        
        # nvidia-smi 정보가 있으면 사용
        if gpu_id in gpu_memory_info:
            free_memory = gpu_memory_info[gpu_id]['free']
            if free_memory >= min_memory_mb:
                available.append(gpu_id)
        else:
            # Fallback: PyTorch 메모리 정보 사용
            try:
                torch.cuda.set_device(gpu_id)
                props = torch.cuda.get_device_properties(gpu_id)
                total = props.total_memory / 1024**2
                reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
                free = total - reserved
                if free >= min_memory_mb:
                    available.append(gpu_id)
            except:
                continue
    
    # 사용 가능한 메모리가 많은 순으로 정렬
    if gpu_memory_info:
        available.sort(key=lambda x: gpu_memory_info.get(x, {}).get('free', 0), reverse=True)
    
    return available

def set_device_auto(min_memory_mb: int = 2000, exclude_gpus: Optional[list] = None, use_multi_gpu: bool = True) -> torch.device:
    """
    자동으로 GPU를 선택하고 device 반환 (multi-GPU 지원)
    
    Args:
        min_memory_mb: 최소 필요한 메모리
        exclude_gpus: 제외할 GPU 리스트
        use_multi_gpu: 여러 GPU 사용 여부
    
    Returns:
        torch.device 객체
    """
    available_gpus = get_available_gpus(min_memory_mb, exclude_gpus)
    
    if not available_gpus:
        device = torch.device("cpu")
        print("사용 가능한 GPU가 없습니다. CPU를 사용합니다.")
        return device
    
    # Multi-GPU 사용
    if use_multi_gpu and len(available_gpus) > 1:
        print(f"사용 가능한 GPU: {len(available_gpus)}개 - {available_gpus}")
        # 첫 번째 GPU를 기본으로 사용하되, DataParallel 사용 가능
        device = torch.device(f"cuda:{available_gpus[0]}")
        torch.cuda.set_device(available_gpus[0])
        gpu_names = [torch.cuda.get_device_name(gid) for gid in available_gpus]
        print(f"Multi-GPU 모드: GPU {available_gpus} 사용")
        for gid, name in zip(available_gpus, gpu_names):
            reserved = torch.cuda.memory_reserved(gid) / 1024**2
            print(f"  GPU {gid} ({name}): {reserved:.0f}MB 사용 중")
    else:
        # Single GPU
        gpu_id = available_gpus[0]
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        gpu_name = torch.cuda.get_device_name(gpu_id)
        
        # 실제 메모리 사용량 확인 (nvidia-smi)
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 3 and int(parts[0].strip()) == gpu_id:
                            memory_used = int(parts[1].strip())
                            memory_total = int(parts[2].strip())
                            print(f"GPU {gpu_id} ({gpu_name}) 선택됨 - 사용 중: {memory_used}MB / {memory_total}MB")
                            break
                else:
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
                    print(f"GPU {gpu_id} ({gpu_name}) 선택됨 - 사용 중: {memory_reserved:.0f}MB")
            else:
                memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
                print(f"GPU {gpu_id} ({gpu_name}) 선택됨 - 사용 중: {memory_reserved:.0f}MB")
        except Exception:
            memory_reserved = torch.cuda.memory_reserved(gpu_id) / 1024**2
            print(f"GPU {gpu_id} ({gpu_name}) 선택됨 - 사용 중: {memory_reserved:.0f}MB")
    
    return device

def wrap_model_multi_gpu(model, available_gpus: List[int]) -> torch.nn.Module:
    """모델을 DataParallel로 감싸서 multi-GPU 사용"""
    if len(available_gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=available_gpus)
        print(f"Model wrapped with DataParallel on GPUs: {available_gpus}")
    return model

