from openai import OpenAI

from evaluator import Evaluator



class TextEvaluator(Evaluator):
    def __init__(self, hf_token, openai_api_key, model_name, judge_model_name="gpt-4o-mini", quantize=False, verbose=False):
        super().__init__(verbose)
        self.client = OpenAI(api_key=openai_api_key)

        self.quantize:bool = quantize
        self.model_name:str = model_name
        self.hf_token:str = hf_token

        self.judge_model_name:str = judge_model_name


    def _generate_llm_responses(self, prompts):
        raise NotImplementedError

    def _get_llm_judge_score(self, responses):
        raise NotImplementedError

    def run(self, prompts):
        responses = self._generate_llm_responses(prompts)

        completion = self.client.chat.completions.create(
            model=self.judge_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ]
        )

        print(completion.choices[0].message.content)


    def print_summary(self):
        return super().print_summary()


    def cleanup(self):
        return super().cleanup()