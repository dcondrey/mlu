import logging

class Accelerators:
    def __init__(self):
        pass

    def use_gpu(self):
        try:
            import torch
            
            if torch.cuda.is_available():
                torch.cuda.init()
                logging.info("GPU has been successfully utilized.")
            else:
                logging.warning("No GPU found. Ensure your system has a CUDA-enabled GPU.")
        except Exception as e:
            logging.error("Failed to utilize GPU: %s", e, exc_info=True)

    def use_tpu(self):
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            
            tpu_cores = xm.xrt_world_size()
            if tpu_cores > 0:
                xm.rendezvous('initialization')  # Ensure TPU cores are synchronized for computation
                xm.mark_step()  # Mark the step for the TPU execution
                logging.info("TPU has been successfully utilized with %d cores.", tpu_cores)
            else:
                logging.warning("No TPU found. Ensure your system has TPU access.")
        except Exception as e:
            logging.error("Failed to utilize TPU: %s", e, exc_info=True)

    def check_gpu_memory(self):
        try:
            import torch
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                logging.info(f"Total GPU memory: {total_memory}")
            else:
                logging.warning("Cannot check GPU memory as no GPU is found.")
        except Exception as e:
            logging.error("Failed to check GPU memory: %s", e, exc_info=True)

    def check_tpu_memory(self):
        try:
            import torch_xla.core.xla_model as xm
            if xm.xrt_world_size() > 0:
                # Note: Direct querying of TPU memory might not be supported in all environments.
                # This is a basic approach to infer TPU memory usage.
                tpu_memory_info = xm.get_memory_info(xm.xla_device())
                logging.info(f"TPU Memory - Total: {tpu_memory_info['kb_total']} KB, Free: {tpu_memory_info['kb_free']} KB")
            else:
                logging.warning("No TPU found. Checking TPU memory requires TPU access.")
        except Exception as e:
            logging.error("Failed to check TPU memory: %s", e, exc_info=True)