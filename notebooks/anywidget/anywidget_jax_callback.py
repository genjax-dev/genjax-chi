# %%
from jax.experimental import io_callback
import jax
import jax.numpy as jnp
import anywidget
import asyncio
from uuid import uuid4


# %%


class Widget(anywidget.AnyWidget):
    _esm = """
  import { html, render, useState, useEffect } from 'https://esm.sh/htm/preact/standalone'
  
  const useCustomMsg = (model) => {
      const [lastMsg, setMsg] = useState()
      const handleMsg = (message, info) => {
          if (message.kind == 'rpc') {
            console.log(message)
            setMsg(message)    
          }
          
      }
      useEffect(() => {
        model.on("msg:custom", handleMsg)
        return () => model.off("msg:custom", handleMsg)    
      }, [model])
      return lastMsg
  }

  async function renderWidget({model, el, experimental}) {
    const {invoke} = experimental
    const App = () => {
        const msg = useCustomMsg(model)
        useEffect(() => {
            if (msg) {
                invoke("_receive_message", {id: msg.id, result: msg.message + " foo"}, [])    
            }
            }, 
        [msg])
        return html`<div>
                    ${msg}
                </div>`
    }
    render(html`<${App}/>`, el)
  }
  export default {render: renderWidget}
  """

    def __init__(self):
        super().__init__()
        self._futures = {}

    def send_message(self, message):
        msg_id = str(uuid4())
        future = asyncio.Future()
        self._futures[msg_id] = future
        self.send({"id": msg_id, "kind": "rpc", "message": message})
        return future

    @anywidget.experimental.command
    def _receive_message(self, msg, buffers):
        print("MSG", msg)
        msg_id = msg["id"]
        result = msg["result"]
        future = self._futures.pop(msg_id)
        future.set_result(result)
        return ["ok", []]


w = Widget()
w
# %%

async def do_something_impure(x):
    """Generate a random array like x using the global_rng state"""
    print(f"Doing something impure with {x}")
    return x


@jax.jit
def numpy_random_like(x):
    return io_callback(do_something_impure, x, x)


jax.vmap(numpy_random_like)(jnp.zeros(2))
