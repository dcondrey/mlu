from mlu.modules.accelerators import Accelerators
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    accelerators = Accelerators()
    try:
        # Demonstrating the usage of the Accelerators module
        logging.info("Utilizing GPU resources...")
        accelerators.use_gpu()
        logging.info("Utilizing TPU resources...")
        accelerators.use_tpu()
        logging.info("Demonstration of Accelerators module usage completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during the Accelerators module demonstration: {e}", exc_info=True)

if __name__ == "__main__":
    main()