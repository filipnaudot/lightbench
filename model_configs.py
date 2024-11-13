#                 MODEL NAME                   QUANT  FEW-SHOT
MODELS = [("meta-llama/Llama-3.2-1B-Instruct", False, False),
          ("meta-llama/Llama-3.2-1B-Instruct", False, True),
          ("meta-llama/Llama-3.2-1B-Instruct", True,  False),
          ("meta-llama/Llama-3.2-1B-Instruct", True,  True),
          ("meta-llama/Llama-3.2-3B-Instruct", False, False),
          ("meta-llama/Llama-3.2-3B-Instruct", False, True),
          ("meta-llama/Llama-3.2-3B-Instruct", True,  False),
          ("meta-llama/Llama-3.2-3B-Instruct", True,  True),
          # TODO: Check if model can run on GPU instead of manual remove
          ("meta-llama/Llama-3.1-8B-Instruct", False, False),
          ("meta-llama/Llama-3.1-8B-Instruct", False, True),
          ("meta-llama/Llama-3.1-8B-Instruct", True,  False),
          ("meta-llama/Llama-3.1-8B-Instruct", True,  True),
          ]