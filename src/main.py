from mlp_sdk.abstract import Task
from mlp_sdk.hosting.host import host_mlp_cloud
from mlp_sdk.transport.MlpServiceSDK import MlpServiceSDK
from pydantic import BaseModel, ValidationError
from fastapi import HTTPException
import requests

API_URL = "https://caila.io/api/adapters/openai/chat/completions"
API_KEY = "1000169780.107212.mRxJYRc5QJQr3PsxRMAT2xY5ZdiukBrGaBuKcFB0"
MODEL_ID = "just-ai/openai-proxy/gpt-4o"


class Span(BaseModel):
    start_index: int
    end_index: int


class Entity(BaseModel):
    value: str
    entity_type: str
    span: Span
    entity: str
    source_type: str = "gpt-4o"


class EntitiesList(BaseModel):
    entities: list[Entity]


class PredictRequest(BaseModel):
    texts: list[str]


class PredictResponse(BaseModel):
    entities_list: list[EntitiesList]


class FioExtractionService(Task):
    def __init__(self, config: BaseModel, service_sdk: MlpServiceSDK = None) -> None:
        super().__init__(config, service_sdk)

    @staticmethod
    def call_llm_model(text: str):

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": MODEL_ID,
            "messages": [{"role": "user",
                          "content": f"Выдели все имена, фамилии и отчества, которые встречаются в тексте, и верни "
                                     f"их в точности в той форме, в которой они написаны в тексте. Не изменяй регистр, "
                                     f"склонение или другие свойства слов. Если в тексте есть хотя бы одно имя, "
                                     f"фамилия или отчество, верни их. Если таких компонентов нет, верни пустую "
                                     f"строку. Не добавляй никакого другого текста, кроме найденных имен, "
                                     f"фамилий или отчеств.: {text}"}],
            "stream": False
        }

        response = requests.post(API_URL, json=data, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Error from LLM API: {response.status_code}, {response.text}")

    @staticmethod
    def process_llm_response(llm_response, text):

        found_entities = []
        content = llm_response['choices'][0]['message']['content'].strip()

        if content:
            words = content.split()
            for word in words:
                if word in text:
                    start_index = text.index(word)
                    end_index = start_index + len(word)
                    entity = Entity(
                        value=word,
                        entity_type="PERSON",
                        span=Span(start_index=start_index, end_index=end_index),
                        entity=word
                    )
                    found_entities.append(entity)
        return found_entities

    def predict(self, data: PredictRequest, config: BaseModel) -> PredictResponse:

        entities_list = []
        for text in data.texts:
            if len(text) > 1000:
                raise ValueError(
                    f"Текст длиной более 1000 символов не поддерживается: текущая длина {len(text)} символов")

            llm_response = self.call_llm_model(text)

            entities = self.process_llm_response(llm_response, text)

            entities_list.append(EntitiesList(entities=entities))
        return PredictResponse(entities_list=entities_list)

    def handle_validation_error(self, exc: ValidationError):
        error_details = {
            "errorCode": "mlp-action.common.processing-exception",
            "message": str(exc)
        }
        raise HTTPException(status_code=422, detail=error_details)


if __name__ == "__main__":
    host_mlp_cloud(FioExtractionService, BaseModel())
