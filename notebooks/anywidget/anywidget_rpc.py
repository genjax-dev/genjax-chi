# %%

# Commented out IPython magic to ensure Python compatibility.
# %pip install anywidget

import anywidget


class Widget(anywidget.AnyWidget):
    _esm = """
  import { html, render, useState } from 'https://esm.sh/htm/preact/standalone'

  function App({invoke, model}){
    const [returned, setReturned] = useState()

    const ping = async () => {
      const t0 = Date.now()
      const [message, buffers] = await invoke("ping", null)
      const elapsed = Date.now() - t0
      setReturned(`${message}, ${elapsed}ms`)
    }
    return html`<div>
      <button onClick=${(e)=>ping()}>Ping</button>
      ${returned}
    </div>`
  }

  async function renderWidget({model, el, experimental}) {
    render(html`<${App} invoke=${experimental.invoke}/>`, el)
  }
  export default {render: renderWidget}
  """

    @anywidget.experimental.command
    def ping(self, msg, buffers):
        return "pong", None


w = Widget()
w
