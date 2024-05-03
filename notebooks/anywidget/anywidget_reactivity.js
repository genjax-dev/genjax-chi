
// using https://vanjs.org for reactive html without React/jsx (no build step!)
import van from "https://cdn.jsdelivr.net/gh/vanjs-org/van/public/van-1.5.0.min.js"



// https://tailwindcss.com
const installTailwind = () => {
    if (!document.getElementById("tailwind-cdn")) {
        const script = document.createElement("script");
        script.id = "tailwind-cdn";
        script.src = "https://cdn.tailwindcss.com";
        document.head.appendChild(script);
    }
}

// html basics
const { button, div, pre, br, span } = van.tags
// MathML
const { math, mi, mn, mo, mrow, msup } = van.tags("http://www.w3.org/1998/Math/MathML")
// svg
const { circle, path, svg } = van.tags("http://www.w3.org/2000/svg")

const Smiley = (faceSize, happiness) => {
    return svg({ width: () => `${faceSize.val}px`, viewBox: "0 0 50 50" },
        circle({ cx: "25", cy: "25", "r": "20", stroke: "black", "stroke-width": "2", fill: "yellow" }),
        circle({ cx: "16", cy: "20", "r": "2", stroke: "black", "stroke-width": "2", fill: "black" }),
        circle({ cx: "34", cy: "20", "r": "2", stroke: "black", "stroke-width": "2", fill: "black" }),
        path({ "d": () => `M 15 30 Q 25 ${20 + (20 * happiness.val)}, 35 30`, stroke: "black", "stroke-width": "2", fill: "transparent" }),
    )
}

const Euler = () => math(
    msup(mi("e"), mrow(mi("i"), mi("π"))), mo("+"), mn("1"), mo("="), mn("0"),
)

// reactive state with 2-way binding to the python AnyClass widget's attribute
const state = (model, attr) => {
    let st = van.state(model.get(attr))

    van.derive(() => {
        // reactively read local state and send changes to python
        if (st.val != model.get(attr)) {
            // console.log(`sending ${attr} to python`)
            model.set(attr, st.val)
            model.save_changes()
        }
        
    })
    model.on(`change:${attr}`, () => {
        // subscribe to changes from python, update local state
        if (st.val != model.get(attr)) {
            // console.log(`receiving ${attr} from python`)
            st.val = model.get(attr)
        }
        
    })
    return st
}

const render = function ({ model, el }) {
    const happiness = state(model, "happiness")
    const faceSize = state(model, "faceSize")  
    const mousePos = state(model, "mousePos")
    const received_messages = van.state([])

    // model.on('msg:custom', (message) => received_messages.val = [message, ...received_messages.val])

    van.add(el, div({
        class: "flex flex-col items-center gap-3 bg-pink-100", 
        onclick: (e) => happiness.val = 0.5,
        onmousemove: (e) => {    
            mousePos.val = { x: e.offsetX, y: e.offsetY, height: e.currentTarget.offsetHeight }
        }},
        span({ style: `font-size: 30px` }, Euler()),
        Smiley(faceSize, happiness),
        () => `happiness: ${happiness.val}`
        ))
}

const simpleRender = function ({ model, el }) {
    const happiness = van.state(true)

    // model.on('msg:custom', (message) => received_messages.val = [message, ...received_messages.val])

    van.add(el, div({
        class: "flex flex-col items-center gap-3 h-40 bg-pink", 
        onclick: (e) => happiness.val = !happiness.val},
        () => `happiness: ${happiness.val}`
        ))
}

export default { initialize: installTailwind, render: render }