class ModelSetupConfigurator:
    def __init__(self):
        self.model_names = [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct"  # TODO: Check if model can run on GPU instead of manual removal
        ]

    def generate_list(self, quantization_settings=False, few_shot_settings=False):
        quant_options = [False, True] if quantization_settings else [False]
        few_shot_options = [False, True] if few_shot_settings else [False]

        models = [
            (model_name, quant, fs)
            for model_name in self.model_names
            for quant in quant_options
            for fs in few_shot_options
        ]
        return models