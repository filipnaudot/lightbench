from openai import OpenAI


class LLMJudge:
    def __init__(self, openai_api_key, judge_model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=openai_api_key)
        self.judge_model_name = judge_model_name

    def get_score(self, prompt, response):
        llm_judge_response = self.client.chat.completions.create(
            model=self.judge_model_name,
            messages=[
                {"role": "system", "content": "You are a text evaluator that evaluates text generated by large language models. \
                                               You will be given the prompt that was given to the large language model (denoted with 'PROMPT') and the answer generated by the model (denoted with 'RESPONSE TO EVALUATE'). \
                                               Label text with a score from 1 to 10. 1 being very bad and 10 being perfect.\
                                               Answer only with one number, nothing else. Just and integer: 1, 2, 3, 4, 5, 6, 7, 8, 9, or 10."},
                {
                    "role": "user",
                    "content": f"PROMPT: {prompt}. RESPONSE TO EVALUATE:\n{response}"
                }
            ],
            max_tokens=1,
        )
        # TODO: Add check to see if response is within restrictions.
        score = llm_judge_response.choices[0].message.content

        return  int(score)