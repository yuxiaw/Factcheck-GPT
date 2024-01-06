import time
import openai
openai.api_key = "" # set openai key here

def gpt_single_try(user_input, model = "gpt-3.5-turbo", system_role = "You are a helpful assistant."):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_input},
        ]
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result


def gpt(user_input, model = "gpt-3.5-turbo", 
        system_role="You are a helpful assistant.", 
        num_retries=3, waiting_time = 1):
    r = ''
    for _ in range(num_retries):
        try:
            r = gpt_single_try(user_input, model, system_role)
            break
        except openai.error.OpenAIError as exception:
            print(f"{exception}. Retrying...")
            time.sleep(waiting_time)
    return r