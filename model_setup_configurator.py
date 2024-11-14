class ModelSetupConfigurator:
    def __init__(self):
        self.model_names = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct"  # TODO: Check if model can run on GPU instead of manual removal
        ]

    def generate_list(self, use_quantization=False, use_few_shot=False):
        quantization_options = [False, True] if use_quantization else [False]
        few_shot_options = [False, True] if use_few_shot else [False]

        models = [
            (model_name, quant, fs)
            for model_name in self.model_names
            for quant in quantization_options
            for fs in few_shot_options
        ]
        return models