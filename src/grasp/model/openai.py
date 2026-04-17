import json
from typing import Any
from uuid import uuid4

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.responses import Response as OpenAIResponse

from grasp.configs import ModelConfig
from grasp.model.base import Message, Model, Reasoning, Response, ToolCall, strip_none


class OpenAICompletionsModel(Model):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = OpenAI(
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
            timeout=config.completion_timeout,
            max_retries=config.num_retries,
        )

    @staticmethod
    def prepare_messages(messages: list[Message]) -> list[dict[str, Any]]:
        msgs = []
        for msg in messages:
            if isinstance(msg.content, str):
                msgs.append(msg.model_dump())
                continue

            # response content
            assert isinstance(msg.content.raw, ChatCompletion)
            msgs.append(msg.content.raw.choices[0].message)
            for tool_call in msg.content.tool_calls:
                assert tool_call.result is not None, "Expected tool call result"
                msgs.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": tool_call.result,
                    }
                )

        return msgs

    def call(
        self,
        messages: list[Message],
        fns: list[dict],
        config: ModelConfig | None = None,
    ) -> Response:
        if config is None:
            config = self.config

        response: ChatCompletion = self.client.chat.completions.create(
            model=config.model,
            messages=self.prepare_messages(messages),  # type: ignore
            tools=[{"type": "function", "function": fn} for fn in fns],  # type: ignore
            tool_choice=config.tool_choice,  # type: ignore
            parallel_tool_calls=config.parallel_tool_calls,
            temperature=config.temperature,
            top_p=config.top_p,
            max_completion_tokens=config.max_completion_tokens,
            **config.model_kwargs,
        )

        # convert completions api response to our response
        id = uuid4().hex
        if not response.choices:
            return Response(id=id, message=None, tool_calls=[], raw=response)

        choice = response.choices[0]  # type: ignore
        if choice.finish_reason not in ["tool_calls", "stop", "length"]:
            raise ValueError(f"Unexpected finish reason {choice.finish_reason}")

        message = strip_none(choice.message.content)
        reasoning = None
        if hasattr(choice.message, "reasoning_content"):
            reasoning = Reasoning(
                id=uuid4().hex,
                content=strip_none(choice.message.reasoning_content),  # type: ignore
            )

        tool_calls = []
        for tool_call in choice.message.tool_calls or []:
            if tool_call.type != "function":
                continue

            assert tool_call.function.name is not None

            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=json.loads(tool_call.function.arguments),
                )
            )

        # Extract logprobs from choice.logprobs.content (standard OpenAI format)
        token_logprobs = None
        token_texts = None
        logprobs = getattr(choice, "logprobs", None)
        if logprobs and getattr(logprobs, "content", None):
            token_logprobs = [entry.logprob for entry in logprobs.content]
            token_texts = [entry.token for entry in logprobs.content]

        # Extract token_ids and prompt_token_ids from provider_specific_fields (vLLM extension)
        token_ids = None
        prompt_token_ids = None
        if (
            hasattr(choice, "provider_specific_fields")
            and choice.provider_specific_fields  # type: ignore
        ):
            token_ids = choice.provider_specific_fields.get("token_ids")  # type: ignore
            prompt_token_ids = choice.provider_specific_fields.get("prompt_token_ids")  # type: ignore
        # Also check response-level attributes
        if token_ids is None and hasattr(response, "token_ids"):
            token_ids = response.token_ids  # type: ignore
        if prompt_token_ids is None and hasattr(response, "prompt_token_ids"):
            prompt_token_ids = response.prompt_token_ids  # type: ignore

        return Response(
            id=id,
            message=message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=response.usage.model_dump(exclude_defaults=True),  # type: ignore
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            token_texts=token_texts,
            raw=response,
        )


class OpenAIResponsesModel(Model):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self.client = OpenAI(
            base_url=config.model_endpoint,
            api_key=config.model_api_key,
            timeout=config.completion_timeout,
            max_retries=config.num_retries,
        )

    @staticmethod
    def prepare_input(messages: list[Message]) -> list[dict[str, Any]]:
        msgs = []

        for msg in messages:
            if isinstance(msg.content, str):
                msgs.append(msg.model_dump())
                continue

            assert isinstance(msg.content.raw, OpenAIResponse)
            msgs.extend(msg.content.raw.output)
            for tool_call in msg.content.tool_calls:
                assert tool_call.result is not None, "Expected tool call result"
                msgs.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.id,
                        "output": tool_call.result,
                    }
                )

        return msgs

    def call(
        self,
        messages: list[Message],
        fns: list[dict],
        config: ModelConfig | None = None,
    ) -> Response:
        if config is None:
            config = self.config

        # use responses API
        response = self.client.responses.create(
            model=config.model,
            input=self.prepare_input(messages),  # type: ignore
            tools=[{"type": "function", **fn} for fn in fns],  # type: ignore
            tool_choice=config.tool_choice,  # type: ignore
            parallel_tool_calls=config.parallel_tool_calls,
            temperature=config.temperature,
            top_p=config.top_p,
            max_output_tokens=config.max_completion_tokens,
            **config.model_kwargs,
            store=False,
            include=["reasoning.encrypted_content", "message.input_image.image_url"],
        )

        message = None
        reasoning = None
        tool_calls = []

        for output in response.output:
            if output.type == "message":
                if output.content:
                    message = "\n\n".join(entry.text for entry in output.content)

            elif output.type == "reasoning":
                # reasoning
                reasoning = Reasoning(id=output.id)
                if output.summary:
                    reasoning.summary = "\n\n".join(
                        entry.text for entry in output.summary
                    )

                if output.content:
                    reasoning.content = "\n\n".join(
                        entry.text for entry in output.content
                    )

                if output.encrypted_content is not None:
                    reasoning.encrypted_content = output.encrypted_content

            elif output.type == "function_call":
                # tool call
                tool_calls.append(
                    ToolCall(
                        id=output.call_id,
                        name=output.name,
                        args=json.loads(output.arguments),
                    )
                )

            else:
                raise ValueError(f"Unknown output type {type(output)}")

        return Response(
            id=response.id,
            message=message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=response.usage.model_dump(exclude_defaults=True),  # type: ignore
            raw=response,
        )
