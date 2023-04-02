import threading
import queue

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.schema import (
    HumanMessage,
    SystemMessage
)

app = FastAPI(
    title="LangChain API",
)


class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)


class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)


def llm_thread(g, prompt):
    try:
        chat = ChatOpenAI(
            verbose=True,
            streaming=True,
            callback_manager=CallbackManager([ChainStreamHandler(g)]),
            temperature=0.7,
        )
        chat([SystemMessage(content="You are a poetic assistant"), HumanMessage(content=prompt)])

    finally:
        g.close()


def chat(prompt):
    g = ThreadedGenerator()
    threading.Thread(target=llm_thread, args=(g, prompt)).start()
    return g


@app.get("/question-stream")
async def stream():
    return StreamingResponse(
        chat("Write me a song about sparkling water."),
        media_type='text/event-stream'
    )


def start():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
