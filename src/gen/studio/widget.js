import {createRender, useModelState} from "https://esm.sh/@anywidget/react@0.0.7"
import * as React from 'https://esm.sh/react@18.3.1'
import * as ReactDOM from 'https://esm.sh/react-dom@18.3.1'
import htm from 'https://esm.sh/htm@3.1.1'
import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.14/+esm"
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.9.0/+esm"

const html = htm.bind(React.createElement)

/**
 * Wrap plot specs so that our node renderer can identify them.
 */
class PlotSpec {
  /**
   * Create a new plot spec.
   */
  constructor(spec) {
    this.spec = spec;
  }

  /**
   * Render the plot.
   */
  plot() {
    return Plot.plot(this.spec);
  }
}

/**
 * Create a new element.
 */
const el = (tag, props, ...children) => {
  if (props.constructor !== Object) {
    children.unshift(props);
    props = {};
  }
  let baseTag
  if (typeof tag === 'string') {
    let id, classes
    [baseTag, ...classes] = tag.split('.');
    [baseTag, id] = baseTag.split('#'); 
    
    if (id) {props.id = id;}
    
    if (classes.length > 0) {
      props.className = `${props.className || ''} ${classes.join(' ')}`.trim();
    }
     
  } else {
    baseTag = tag
  }
  children = children.map((child) => html`<${Node} value=${child} />`)
  return html`<${baseTag} ...${props}>${children}</${tag}>`;
};


const scope = {
  d3,
  Plot, React, ReactDOM,
  View: {
    Plot: (x) => new PlotSpec(x),
    el
  }  
}
const { useState, useEffect, useRef, useCallback } = React

/**
 * Interpret data recursively, evaluating functions.
 */
export function interpret(data) {
  if (data === null) return null;
  if (Array.isArray(data)) return data.map(interpret);
  if (typeof data === "string" || data instanceof String) return data;
  if (Object.entries(data).length == 0) return data;
  
  switch (data["pyobsplot-type"]) {
    case "function":
      let fn = data.name ? scope[data.module][data.name] : scope[data.module]
      return fn.call(null, ...interpret(data["args"]));
    case "ref":
      return data.name ? scope[data.module][data.name] : scope[data.module]
    case "js":
      // Use indirect eval to avoid bundling issues
      // See https://esbuild.github.io/content-types/#direct-eval
      let indirect_eval = eval;
      return indirect_eval(data["value"]);
    case "datetime":
      return new Date(data["value"]);
  }
  // recurse into objects
  let ret = {};
  for (const [key, value] of Object.entries(data)) {
    ret[key] = interpret(value);
  }
  return ret;
}

/**
 * Renders a plot.
 */
function PlotView({ spec }) {
  const [parent, setParent] = useState(null)
  const ref = useCallback(setParent)
  useEffect(() => {
    if (parent) {
      const plot = spec.plot()
      parent.appendChild(plot)
      return () => parent.removeChild(plot)
    }
  }, [spec, parent])
  return html`<div ref=${ref}></div>`
}

/**
 * Renders a node. Arrays are parsed as hiccup.
 */
function Node({ value }) {
  if (Array.isArray(value)) {
    return el.apply(null, value);
  } else if (value instanceof PlotSpec) {
    return html`<${PlotView} spec=${value}/>`;
  } else {
    return value
  }
}

/**
 * The main app.
 */
function App() { 
  const [data, _] = useModelState("data")
  const value = data ? interpret(JSON.parse(data)) : null
  return html`<div class="pa1"><${Node} value=${value}/></div>`
}

const render = createRender(App)

/**
 * Install Tachyons CSS.
 */
const installTachyons = () => {
  const id = "tachyons-cdn"
  const url = "https://unpkg.com/tachyons@4.12.0/css/tachyons.min.css"
  if (!document.getElementById(id)) {
      const link = document.createElement("link");
      link.id = id;
      link.rel = "stylesheet";
      link.href = url;
      document.head.appendChild(link);
  }
}

export default { render, initialize: installTachyons }