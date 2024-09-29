import{_ as n,c as a,a0 as p,o as e}from"./chunks/framework.DOEc3Y8C.js";const r=JSON.parse('{"title":"pattern matching","description":"","frontmatter":{},"headers":[],"relativePath":"language/rust/10.pattern-matching.md","filePath":"language/rust/10.pattern-matching.md"}'),t={name:"language/rust/10.pattern-matching.md"};function l(i,s,o,c,d,g){return e(),a("div",null,s[0]||(s[0]=[p(`<h1 id="pattern-matching" tabindex="-1">pattern matching <a class="header-anchor" href="#pattern-matching" aria-label="Permalink to &quot;pattern matching&quot;">​</a></h1><p>模式匹配是 Rust 中的一个功能强大的特性，允许你检查一个值的结构并据此执行代码。模式匹配经常与 match 表达式和 if let 结构一起使用。</p><p>以下是模式匹配的一些关键点：</p><p><strong>基本匹配</strong></p><p>最简单的模式匹配形式是 match 表达式。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let x = 5;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>match x {</span></span>
<span class="line"><span>    1 =&gt; println!(&quot;one&quot;),</span></span>
<span class="line"><span>    2 =&gt; println!(&quot;two&quot;),</span></span>
<span class="line"><span>    3 =&gt; println!(&quot;three&quot;),</span></span>
<span class="line"><span>    4 =&gt; println!(&quot;four&quot;),</span></span>
<span class="line"><span>    5 =&gt; println!(&quot;five&quot;),</span></span>
<span class="line"><span>    _ =&gt; println!(&quot;something else&quot;),</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在上述代码中，x 的值与每个模式进行匹配，当找到匹配项时，执行相应的代码。下划线 _ 是一个通配符模式，匹配任何值。</p><p><strong>解构/拆分</strong></p><p>模式可以用来解构结构体、枚举、元组和引用。</p><p>模式匹配不仅可以用于简单的值比较，还可以用于拆解（或解构）更复杂的数据类型，从而让你可以直接访问其内部的值。这使得模式匹配成为一个非常强大的工具，尤其是在处理复杂数据结构时。</p><ul><li><strong>解构结构体</strong>:</li></ul><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>struct Point {</span></span>
<span class="line"><span>    x: i32,</span></span>
<span class="line"><span>    y: i32,</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let p = Point { x: 3, y: 4 };</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>match p {</span></span>
<span class="line"><span>    Point { x, y } =&gt; println!(&quot;x: {}, y: {}&quot;, x, y),</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在上面的代码中，<code>Point { x, y }</code> 是一个模式，它匹配任何 <code>Point</code> 结构体，并解构它，将其字段绑定到变量 <code>x</code> 和 <code>y</code>。</p><ul><li><strong>解构枚举</strong>:</li></ul><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>enum Option&lt;T&gt; {</span></span>
<span class="line"><span>    Some(T),</span></span>
<span class="line"><span>    None,</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let some_value = Some(5);</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>match some_value {</span></span>
<span class="line"><span>    Some(num) =&gt; println!(&quot;Got a number: {}&quot;, num),</span></span>
<span class="line"><span>    None =&gt; println!(&quot;Got nothing&quot;),</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在这里，<code>Some(num)</code> 是一个模式，它匹配 <code>Some</code> 变体并解构其内部值。</p><ul><li><strong>解构元组</strong>:</li></ul><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let tuple = (1, &quot;hello&quot;);</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>match tuple {</span></span>
<span class="line"><span>    (a, b) =&gt; println!(&quot;Got: {} and {}&quot;, a, b),</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>这里，<code>(a, b)</code> 是一个模式，它匹配任何两元素的元组，并解构它。</p><p><strong>以引用方式匹配</strong></p><p>在 Rust 中，当你使用模式匹配对引用进行匹配时，你可能需要考虑两个方面：匹配引用本身，以及匹配引用指向的值。这时，你可以结合使用模式和解引用来达到目的。</p><ul><li><strong>匹配引用</strong>：</li></ul><p>直接匹配引用本身。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let x = &amp;5;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>match x {</span></span>
<span class="line"><span>    &amp;v =&gt; println!(&quot;x is a reference to {}&quot;, v),</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在这个例子中，我们使用 <code>&amp;v</code> 的模式匹配引用，并将其解构为 <code>v</code>。</p><ul><li><strong>使用解引用和匹配</strong>：</li></ul><p>使用 <code>*</code> 进行解引用，并与模式匹配结合使用。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let x = &amp;5;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>match *x {</span></span>
<span class="line"><span>    v =&gt; println!(&quot;x is referencing {}&quot;, v),</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>这里，我们先使用 <code>*x</code> 进行解引用，然后使用模式 <code>v</code> 进行匹配。</p><ul><li><strong>在 <code>match</code> 中使用解引用</strong>：</li></ul><p>在 <code>match</code> 表达式中，你可以直接使用解引用的模式。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let x = &amp;5;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>match x {</span></span>
<span class="line"><span>    &amp;v =&gt; println!(&quot;x is a reference to {}&quot;, v),</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>注意，这与第一个例子相同，但在更复杂的模式中，这种方式可能更有用。</p><p><strong>if let语句</strong></p><p>解释下：&quot;语法糖&quot;（Syntactic sugar）是编程语言中的一个术语，指的是那些没有引入新的功能，但可以让代码更易读或更简洁的语法。语法糖的存在是为了使编程更加方便，使代码更加直观和易于理解。</p><p>换句话说，语法糖是一种便捷的编写方式，它只是现有功能的另一种表示。在编译或解释代码时，这种语法通常会被转换为更基础的、标准的语法。</p><p><code>if let</code> 语句是 Rust 中的一个语法糖，它允许你结合 <code>if</code> 语句和模式匹配。它特别适用于当你只关心一种匹配情况，并想在这种情况下执行某个代码块时。</p><p><code>if let</code> 的基本形式如下：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>if let PATTERN = EXPRESSION {</span></span>
<span class="line"><span>    // 代码块</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>其中，<code>PATTERN</code> 是你想匹配的模式，<code>EXPRESSION</code> 是你想匹配的表达式。</p><p>让我们看一些实际的例子：</p><ol><li><p><strong>匹配 <code>Option</code></strong>：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let some_value = Some(5);</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>if let Some(x) = some_value {</span></span>
<span class="line"><span>    println!(&quot;Got a value: {}&quot;, x);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>这里，只有当 <code>some_value</code> 是 <code>Some</code> 变体时，代码块才会执行。<code>x</code> 被绑定到 <code>Some</code> 内的值，并在代码块中使用。</p></li><li><p><strong>与 <code>else</code> 结合使用</strong>：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let some_value = None;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>if let Some(x) = some_value {</span></span>
<span class="line"><span>    println!(&quot;Got a value: {}&quot;, x);</span></span>
<span class="line"><span>} else {</span></span>
<span class="line"><span>    println!(&quot;Didn&#39;t match Some&quot;);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>如果 <code>if let</code> 的模式不匹配，你可以使用 <code>else</code> 分支。</p></li><li><p><strong>匹配枚举</strong>：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>enum Message {</span></span>
<span class="line"><span>    Hello(String),</span></span>
<span class="line"><span>    Bye,</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let msg = Message::Hello(String::from(&quot;World&quot;));</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>if let Message::Hello(s) = msg {</span></span>
<span class="line"><span>    println!(&quot;Hello, {}&quot;, s);</span></span>
<span class="line"><span>}</span></span></code></pre></div></li></ol><p>这里，我们只在 <code>msg</code> 是 <code>Message::Hello</code> 变体时执行代码块。</p><p><code>if let</code> 的主要优势是它提供了一个简洁的方式来处理只关心的一种匹配情况，而无需编写完整的 <code>match</code> 语句。</p><p>毕竟是语法糖，if let 和match用法是一样的，区别就在于if let更为精简</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>if let Some(x) = some_option {</span></span>
<span class="line"><span>    println!(&quot;Got a value: {}&quot;, x);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>这个 <code>if let</code> 语句可以用 <code>match</code> 语句重写为：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>atch some_option {</span></span>
<span class="line"><span>    Some(x) =&gt; println!(&quot;Got a value: {}&quot;, x),</span></span>
<span class="line"><span>    _ =&gt; {}</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在这个 <code>match</code> 版本中，我们显式地处理了 <code>Some(x)</code> 和所有其他可能的模式（使用 <code>_</code> 通配符）。但是，<code>if let</code> 提供了一种更简洁的方式来处理我们关心的特定模式，而忽略其他所有模式。</p><p><strong>while let语句</strong></p><p>while let 是 Rust 中的另一个结合了模式匹配和循环的语法糖。它允许你在某个模式匹配成功的情况下持续执行循环体。只要模式匹配成功，循环就会继续；一旦模式匹配失败，循环就会停止。</p><p>while let 的基本结构如下：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>while let PATTERN = EXPRESSION {</span></span>
<span class="line"><span>    // 代码块</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>其中，PATTERN 是你想匹配的模式，而 EXPRESSION 是你想匹配的表达式。</p><p>让我们看一些实际的例子：</p><p><strong>使用 Option：</strong></p><p>假设我们有一个 Vec 并使用 pop 方法。pop 返回一个 Option：如果 Vec 为空，它返回 None；否则，它返回 Some(T)，其中 T 是 Vec 的最后一个元素。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let mut stack = vec![1, 2, 3, 4, 5];</span></span>
<span class="line"><span></span></span>
<span class="line"><span>while let Some(top) = stack.pop() {</span></span>
<span class="line"><span>    println!(&quot;Popped value: {}&quot;, top);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>这里，while let 循环会持续执行，直到 stack.pop() 返回 None，即 stack 为空。</p><p><strong>解构枚举：</strong></p><p>假设我们有一个表示事件的枚举，我们想从一个队列中取出并处理这些事件，直到遇到特定的事件。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>enum Event {</span></span>
<span class="line"><span>    Continue(i32),</span></span>
<span class="line"><span>    Stop,</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let mut events = vec![Event::Continue(5), Event::Continue(10), Event::Stop, Event::Continue(15)];</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>while let Some(Event::Continue(value)) = events.pop() {</span></span>
<span class="line"><span>    println!(&quot;Got a continue event with value: {}&quot;, value);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>这里，while let 循环会处理 Event::Continue 事件，直到遇到不是 Event::Continue 的事件或 events 为空。</p><p><strong>内部绑定</strong></p><p>在 Rust 中，当我们谈论模式匹配时，有时我们会遇到需要访问内部值的情况。内部绑定允许我们在一个模式中同时匹配一个值的结构和捕获其内部值。</p><p>内部绑定的主要用途是在一个模式中匹配一个值，然后在后续的代码中使用该值。</p><p>考虑以下例子：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>enum Message {</span></span>
<span class="line"><span>    Hello { id: i32 },</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let msg = Message::Hello { id: 5 };</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>match msg {</span></span>
<span class="line"><span>    Message::Hello { id: inner_id } if inner_id &gt; 5 =&gt; {</span></span>
<span class="line"><span>        println!(&quot;Hello with an id greater than 5! Got: {}&quot;, inner_id);</span></span>
<span class="line"><span>    },</span></span>
<span class="line"><span>    Message::Hello { id: _ } =&gt; {</span></span>
<span class="line"><span>        println!(&quot;Hello with some id!&quot;);</span></span>
<span class="line"><span>    },</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在上述代码中，我们使用内部绑定来捕获 <code>id</code> 字段的值并将其绑定到 <code>inner_id</code> 变量。然后，我们可以在 <code>match</code> 的分支体中使用这个 <code>inner_id</code>。</p><p>这种内部绑定的能力使得我们可以在模式中进行更复杂的操作，比如在 <code>if</code> 守卫中检查捕获的值。</p><p>总的来说，内部绑定是 Rust 中模式匹配的一个强大特性，允许我们在匹配值的结构的同时，捕获并在后续的代码中使用其内部的值。</p><p><strong>模式匹配的穷尽性</strong></p><p>在 Rust 中，模式匹配的穷尽性（Exhaustiveness）是指必须处理所有可能的情况，确保没有遗漏。这是 Rust 的一个重要特性，因为它确保了代码的健壮性和安全性。</p><p>当使用 <code>match</code> 语句进行模式匹配时，Rust 编译器会检查所有可能的模式是否都被考虑到了。如果有遗漏，编译器会报错。</p><p>例如，考虑一个简单的 <code>enum</code>：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>enum Color {</span></span>
<span class="line"><span>    Red,</span></span>
<span class="line"><span>    Green,</span></span>
<span class="line"><span>    Blue,</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>当我们使用 <code>match</code> 语句进行模式匹配时，我们必须处理所有的 <code>Color</code> 变体：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let my_color = Color::Green;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>match my_color {</span></span>
<span class="line"><span>    Color::Red =&gt; println!(&quot;It&#39;s red!&quot;),</span></span>
<span class="line"><span>    Color::Green =&gt; println!(&quot;It&#39;s green!&quot;),</span></span>
<span class="line"><span>    Color::Blue =&gt; println!(&quot;It&#39;s blue!&quot;),</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>如果我们遗漏了任何一个变体，例如：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>match my_color {</span></span>
<span class="line"><span>    Color::Red =&gt; println!(&quot;It&#39;s red!&quot;),</span></span>
<span class="line"><span>    Color::Green =&gt; println!(&quot;It&#39;s green!&quot;),</span></span>
<span class="line"><span>    // Color::Blue 没有被处理</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>Rust 编译器会报错，因为不是所有的情况都被考虑到了。</p><p>此外，Rust 提供了一个 <code>_</code> 通配符，它可以匹配任何值。这在我们不关心某些模式时非常有用。但要小心使用，确保不要意外地忽略了重要的模式。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>match my_color {</span></span>
<span class="line"><span>Color::Red =&gt; println!(&quot;It&#39;s red!&quot;),</span></span>
<span class="line"><span>    _ =&gt; println!(&quot;It&#39;s some other color!&quot;),</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在这个例子中，除了 <code>Red</code> 以外的所有颜色都会匹配 <code>_</code> 模式。</p><p><strong>模式匹配的使用场合</strong></p><ol><li><p><strong>匹配枚举变体</strong>：模式匹配经常用于处理 <code>enum</code> 类型，因为你可以轻松地区分不同的变体并处理它们。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>enum Message {</span></span>
<span class="line"><span>    Quit,</span></span>
<span class="line"><span>    Move { x: i32, y: i32 },</span></span>
<span class="line"><span>    Write(String),</span></span>
<span class="line"><span>    ChangeColor(i32, i32, i32),</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>match msg {</span></span>
<span class="line"><span>    Message::Quit =&gt; println!(&quot;The Quit variant&quot;),</span></span>
<span class="line"><span>    Message::Move { x, y } =&gt; println!(&quot;Move in the x: {} y: {}&quot;, x, y),</span></span>
<span class="line"><span>    Message::Write(s) =&gt; println!(&quot;Text message: {}&quot;, s),</span></span>
<span class="line"><span>    Message::ChangeColor(r, g, b) =&gt; println!(&quot;Change color to red: {}, green: {}, blue: {}&quot;, r, g, b),</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>解构结构体和元组</strong>：可以使用模式匹配来解构和访问结构体或元组的值。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let point = (3, 5);</span></span>
<span class="line"><span>match point {</span></span>
<span class="line"><span>    (0, 0) =&gt; println!(&quot;Origin&quot;),</span></span>
<span class="line"><span>    (x, 0) =&gt; println!(&quot;On the x-axis at x = {}&quot;, x),</span></span>
<span class="line"><span>    (0, y) =&gt; println!(&quot;On the y-axis at y = {}&quot;, y),</span></span>
<span class="line"><span>    (x, y) =&gt; println!(&quot;Other point at x = {}, y = {}&quot;, x, y),</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>解构引用</strong>：当你处理引用时，模式匹配允许你同时匹配和解引用值。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>match &amp;some_value {</span></span>
<span class="line"><span>    &amp;Some(x) =&gt; println!(&quot;Got a value: {}&quot;, x),</span></span>
<span class="line"><span>    &amp;None =&gt; println!(&quot;No value&quot;),</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>处理 <code>Option</code> 和 <code>Result</code> 类型</strong>：这两种类型经常与模式匹配一起使用，使得处理可能的错误或缺失值变得简单明了。</p></li><li><p><strong><code>if let</code> 和 <code>while let</code> 表达式</strong>：这两种表达式都是基于模式匹配的，它们允许你在特定情况下简化代码。</p></li><li><p><strong>使用守卫进行条件匹配</strong>：你可以在模式匹配中加入额外的条件，使匹配更加灵活。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let num = Some(4);</span></span>
<span class="line"><span>match num {</span></span>
<span class="line"><span>    Some(x) if x &lt; 5 =&gt; println!(&quot;less than five: {}&quot;, x),</span></span>
<span class="line"><span>    Some(x) =&gt; println!(&quot;{}&quot;, x),</span></span>
<span class="line"><span>    None =&gt; (),</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>匹配字面值</strong>：你可以直接匹配特定的字面值，如数字、字符或字符串</p></li></ol><p><strong>for循环模式匹配</strong></p><p>在 Rust 中，<code>for</code> 循环可以与模式匹配结合使用，从而允许你在迭代集合时对每个元素进行解构。这在处理复杂的数据结构时非常有用。</p><p>以下是一些 <code>for</code> 循环与模式匹配结合使用的例子：</p><ol><li><p><strong>迭代元组的数组</strong>：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let points = [(1, 2), (3, 4), (5, 6)];</span></span>
<span class="line"><span>for (x, y) in points.iter() {</span></span>
<span class="line"><span>    println!(&quot;x: {}, y: {}&quot;, x, y);</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>迭代枚举的向量</strong>：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>enum Message {</span></span>
<span class="line"><span>    Move { x: i32, y: i32 },</span></span>
<span class="line"><span>    Write(String),</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let messages = vec![</span></span>
<span class="line"><span>    Message::Move { x: 1, y: 2 },</span></span>
<span class="line"><span>    Message::Write(&quot;Hello&quot;.to_string()),</span></span>
<span class="line"><span>];</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>for message in messages.iter() {</span></span>
<span class="line"><span>    match message {</span></span>
<span class="line"><span>        Message::Move { x, y } =&gt; println!(&quot;Move to x={}, y={}&quot;, x, y),</span></span>
<span class="line"><span>        Message::Write(text) =&gt; println!(&quot;Text message: {}&quot;, text),</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>迭代哈希映射</strong>：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>use std::collections::HashMap;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let mut scores = HashMap::new();</span></span>
<span class="line"><span>scores.insert(&quot;Alice&quot;, 10);</span></span>
<span class="line"><span>scores.insert(&quot;Bob&quot;, 15);</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>for (name, score) in &amp;scores {</span></span>
<span class="line"><span>    println!(&quot;{}: {}&quot;, name, score);</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>解构复杂的结构体</strong>：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>struct Point {</span></span>
<span class="line"><span>    x: i32,</span></span>
<span class="line"><span>    y: i32,</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let points = vec![Point { x: 1, y: 2 }, Point { x: 3, y: 4 }];</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>for Point { x, y } in points.iter() {</span></span>
<span class="line"><span>    println!(&quot;x: {}, y: {}&quot;, x, y);</span></span>
<span class="line"><span>}</span></span></code></pre></div></li></ol><p><strong>函数参数的模式匹配</strong></p><p>在 Rust 中，函数参数也可以使用模式进行匹配。这允许我们在函数签名中直接解构复杂的数据结构，简化函数体中的代码。以下是一些使用模式匹配的函数参数的例子：</p><ol><li><p><strong>解构元组</strong>:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fn print_coordinates((x, y): (i32, i32)) {</span></span>
<span class="line"><span>    println!(&quot;x: {}, y: {}&quot;, x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>print_coordinates((3, 4));  // 输出: x: 3, y: 4</span></span></code></pre></div></li><li><p><strong>解构枚举</strong>:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>enum Message {</span></span>
<span class="line"><span>    Move { x: i32, y: i32 },</span></span>
<span class="line"><span>    Write(String),</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>fn handle_message(msg: Message) {</span></span>
<span class="line"><span>    match msg {</span></span>
<span class="line"><span>        Message::Move { x, y } =&gt; println!(&quot;Move to x={}, y={}&quot;, x, y),</span></span>
<span class="line"><span>        Message::Write(text) =&gt; println!(&quot;Text message: {}&quot;, text),</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>解构结构体</strong>:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>struct Point {</span></span>
<span class="line"><span>    x: i32,</span></span>
<span class="line"><span>    y: i32,</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>fn print_point(Point { x, y }: Point) {</span></span>
<span class="line"><span>    println!(&quot;x: {}, y: {}&quot;, x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let p = Point { x: 5, y: 7 };</span></span>
<span class="line"><span>print_point(p);  // 输出: x: 5, y: 7</span></span></code></pre></div></li><li><p><strong>忽略某些值</strong>:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fn print_first((x, _): (i32, i32)) {</span></span>
<span class="line"><span>    println!(&quot;x: {}&quot;, x);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>print_first((8, 9));  // 输出: x: 8</span></span></code></pre></div></li></ol><p>使用模式匹配的函数参数可以让我们更直接地访问数据的内部结构，而无需在函数体中进行额外的解构。这使得代码更加简洁和直观。</p>`,94)]))}const h=n(t,[["render",l]]);export{r as __pageData,h as default};
