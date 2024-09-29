import{_ as n,c as a,a0 as p,o as e}from"./chunks/framework.DDvRCNFD.js";const h=JSON.parse('{"title":"statement, macro","description":"","frontmatter":{},"headers":[],"relativePath":"language/rust/5.statement-macro.md","filePath":"language/rust/5.statement-macro.md"}'),t={name:"language/rust/5.statement-macro.md"};function l(i,s,o,c,d,r){return e(),a("div",null,s[0]||(s[0]=[p(`<h1 id="statement-macro" tabindex="-1">statement, macro <a class="header-anchor" href="#statement-macro" aria-label="Permalink to &quot;statement, macro&quot;">​</a></h1><p><strong>语句，宏</strong></p><h4 id="条件语句" tabindex="-1">条件语句 <a class="header-anchor" href="#条件语句" aria-label="Permalink to &quot;条件语句&quot;">​</a></h4><p>主要使用<code>if</code>, <code>else if</code>, 和 <code>else</code> 关键字来执行。这与许多其他编程语言的结构相似。</p><p><strong>基本的<code>if</code>语句</strong></p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>if condition {</span></span>
<span class="line"><span>    // 代码块</span></span>
<span class="line"><span>}</span></span></code></pre></div><p><strong><code>if-else</code>语句</strong></p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>if condition {</span></span>
<span class="line"><span>    // 代码块</span></span>
<span class="line"><span>} else {</span></span>
<span class="line"><span>    // 其他代码块</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>使用<code>else if</code>的多条件</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>if condition1 {</span></span>
<span class="line"><span>    // 代码块1</span></span>
<span class="line"><span>} else if condition2 {</span></span>
<span class="line"><span>    // 代码块2</span></span>
<span class="line"><span>} else {</span></span>
<span class="line"><span>    // 其他代码块</span></span>
<span class="line"><span>}</span></span></code></pre></div><p><strong>示例</strong>:</p><p>假设我们有一个变量<code>number</code>，我们想根据它的值打印不同的消息。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let number = 10;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>if number &lt; 10 {</span></span>
<span class="line"><span>    println!(&quot;数字小于10&quot;);</span></span>
<span class="line"><span>} else if number == 10 {</span></span>
<span class="line"><span>    println!(&quot;数字等于10&quot;);</span></span>
<span class="line"><span>} else {</span></span>
<span class="line"><span>    println!(&quot;数字大于10&quot;);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在上面的示例中，输出将是 &quot;数字等于10&quot;。</p><p>此外，Rust中的<code>if</code>语句也可以有一个返回值，这意味着你可以将其结果直接赋值给一个变量。例如:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let condition = true;</span></span>
<span class="line"><span>let number = if condition { 5 } else { 6 };</span></span></code></pre></div><p>在上面的代码中，如果<code>condition</code>为<code>true</code>，则<code>number</code>的值为5，否则为6。</p><p>与c++不同的地方在于，条件部分不需要用小括号引用括起来</p><p>整个条件语句是当作一个表达式来求值的，因此每一个分支都必须是相同类型的表达式。当然，如果作为普通的条件语句来使用的话，可以令类型是（）</p><p>例：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>if x  &lt;= 0 {</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>println!(&quot;too small!&quot;);</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span></code></pre></div><h4 id="循环语句" tabindex="-1">循环语句 <a class="header-anchor" href="#循环语句" aria-label="Permalink to &quot;循环语句&quot;">​</a></h4><p>rust循环主要有3种；while，loop，for</p><ul><li><code>break</code>和<code>continue</code>用于改变循环中的控制流</li></ul><p><strong><code>while</code>循环: 当指定的条件为真时，这个循环会一直执行。</strong></p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let mut number = 5;</span></span>
<span class="line"><span>while number &gt; 0 {</span></span>
<span class="line"><span>    println!(&quot;number 的值是: {}&quot;, number);</span></span>
<span class="line"><span>    number -= 1;</span></span>
<span class="line"><span>}</span></span></code></pre></div><p><strong><code>loop</code> 循环: 这是一个无限循环，除非使用 <code>break</code> 关键字退出。</strong></p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>loop {</span></span>
<span class="line"><span>    println!(&quot;这是一个无限循环&quot;);</span></span>
<span class="line"><span>    // 使用 break 退出循环</span></span>
<span class="line"><span>    if some_condition {</span></span>
<span class="line"><span>        break;</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span></code></pre></div><ul><li><strong>使用 <code>break</code> 退出循环</strong></li></ul><p>这个示例中，当<code>number</code>等于<code>4</code>时，循环会退出。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>for number in 1..10 {</span></span>
<span class="line"><span>    if number == 4 {</span></span>
<span class="line"><span>        println!(&quot;找到了4，退出循环！&quot;);</span></span>
<span class="line"><span>        break;</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    println!(&quot;当前的数字是: {}&quot;, number);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>输出:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>code当前的数字是: 1</span></span>
<span class="line"><span>当前的数字是: 2</span></span>
<span class="line"><span>当前的数字是: 3</span></span>
<span class="line"><span>找到了4，退出循环！</span></span></code></pre></div><p><strong><code>for</code> 循环: 这是一个遍历范围、迭代器或集合的循环。</strong></p><p>a. <strong>范围</strong>:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>for number in 1..4 { // 1到3（不包括4）</span></span>
<span class="line"><span>    println!(&quot;number 的值是: {}&quot;, number);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>b. <strong>数组和切片</strong>:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let arr = [1, 2, 3, 4, 5];</span></span>
<span class="line"><span>for element in arr.iter() {</span></span>
<span class="line"><span>    println!(&quot;数组元素: {}&quot;, element);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>c. <strong>Vector</strong>:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let vec = vec![1, 2, 3, 4, 5];</span></span>
<span class="line"><span>for element in &amp;vec {</span></span>
<span class="line"><span>    println!(&quot;Vector元素: {}&quot;, element);</span></span>
<span class="line"><span>}</span></span></code></pre></div><ol><li><strong><code>for</code> 循环与 <code>enumerate()</code></strong>: 当你需要遍历集合并同时获取每个元素的索引时，这很有用。</li></ol><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let arr = [&quot;a&quot;, &quot;b&quot;, &quot;c&quot;];</span></span>
<span class="line"><span>for (index, value) in arr.iter().enumerate() {</span></span>
<span class="line"><span>    println!(&quot;索引 {}: 值 {}&quot;, index, value);</span></span>
<span class="line"><span>}</span></span></code></pre></div><ol><li><code>continue</code> 和 <code>break</code> 关键字: <ul><li><code>continue</code>: 跳过当前迭代，并继续循环的下一次迭代。</li><li><code>break</code>: 完全退出循环。</li></ul></li></ol><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>for number in 1..6 {</span></span>
<span class="line"><span>    if number == 3 {</span></span>
<span class="line"><span>        continue; // 当 number 为 3 时，跳过并继续下一个迭代</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    if number == 5 {</span></span>
<span class="line"><span>        break; // 当 number 为 5 时，退出循环</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    println!(&quot;number 的值是: {}&quot;, number);</span></span>
<span class="line"><span>}</span></span></code></pre></div><h4 id="print-和println" tabindex="-1"><code>print!</code>和<code>println!</code> <a class="header-anchor" href="#print-和println" aria-label="Permalink to &quot;\`print!\`和\`println!\`&quot;">​</a></h4><p>在Rust中，<code>print!</code> 和 <code>println!</code> 是两个常用的宏，用于向控制台输出文本。它们的行为与许多其他编程语言中的 <code>print</code> 和 <code>println</code> 或 <code>printf</code> 功能类似，但有一些特定的差异。</p><ol><li><strong><code>print!</code> 宏</strong>: <ul><li><code>print!</code> 将文本输出到控制台，但不在其后添加新行。</li><li>语法: <code>print!(&quot;格式字符串&quot;, 参数1, 参数2, ...)</code></li></ul></li><li><strong><code>println!</code> 宏</strong>: <ul><li><code>println!</code> 也将文本输出到控制台，但它会在文本后面添加一个新行，所以随后的输出会从新的一行开始。</li><li>语法: <code>println!(&quot;格式字符串&quot;, 参数1, 参数2, ...)</code></li></ul></li></ol><p><strong>示例</strong>:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>rustCopy codefn main() {</span></span>
<span class="line"><span>    print!(&quot;这是&quot;);</span></span>
<span class="line"><span>    print!(&quot;一个&quot;);</span></span>
<span class="line"><span>    print!(&quot;例子&quot;);</span></span>
<span class="line"><span>    </span></span>
<span class="line"><span>    println!(); // 添加一个新行</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>    println!(&quot;这是&quot;);</span></span>
<span class="line"><span>    println!(&quot;另一个&quot;);</span></span>
<span class="line"><span>    println!(&quot;例子&quot;);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>输出:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>这是一个例子</span></span>
<span class="line"><span>这是</span></span>
<span class="line"><span>另一个</span></span>
<span class="line"><span>例子</span></span></code></pre></div><p>与其他语言的格式化输出功能类似，你可以在字符串中使用占位符，然后在宏调用中传递要插入这些占位符位置的值。例如:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fn main() {</span></span>
<span class="line"><span>    let name = &quot;Alice&quot;;</span></span>
<span class="line"><span>    let age = 30;</span></span>
<span class="line"><span>    println!(&quot;我的名字是{}，我{}岁了。&quot;, name, age);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>输出:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>我的名字是Alice，我30岁了。</span></span></code></pre></div><p>请注意，由于这些都是宏而不是函数，所以在宏名称后面有一个感叹号 (<code>!</code>)。</p><h4 id="format" tabindex="-1">format！ <a class="header-anchor" href="#format" aria-label="Permalink to &quot;format！&quot;">​</a></h4><p>使用与<code>print！</code>/<code>println！</code>相同的用法来创建<code>String</code>字符串。</p><p>format! 宏在Rust中用于创建格式化的字符串，但与print!和println!不同，它不会将结果输出到控制台。相反，它返回一个表示格式化字符串的String。这允许您在其他地方使用或存储格式化的字符串。</p><p>语法:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let formatted_string = format!(&quot;格式字符串&quot;, 参数1, 参数2, ...);</span></span></code></pre></div><p>示例:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let name = &quot;Alice&quot;;</span></span>
<span class="line"><span>let age = 30;</span></span>
<span class="line"><span>let formatted = format!(&quot;我的名字是{}，我{}岁了。&quot;, name, age);</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// \`formatted\` 现在包含 &quot;我的名字是Alice，我30岁了。&quot;</span></span>
<span class="line"><span>println!(&quot;{}&quot;, formatted);</span></span></code></pre></div><p>使用位置参数:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let formatted = format!(&quot;{0} 和 {1} 喜欢 {2}&quot;, &quot;Alice&quot;, &quot;Bob&quot;, &quot;巧克力&quot;);</span></span>
<span class="line"><span>// 输出: &quot;Alice 和 Bob 喜欢 巧克力&quot;</span></span>
<span class="line"><span>println!(&quot;{}&quot;, formatted);</span></span></code></pre></div><p>使用命名参数:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let formatted = format!(&quot;{name} 是一个 {job}&quot;, name=&quot;Alice&quot;, job=&quot;工程师&quot;);</span></span>
<span class="line"><span>// 输出: &quot;Alice 是一个 工程师&quot;</span></span>
<span class="line"><span>println!(&quot;{}&quot;, formatted);</span></span></code></pre></div><p>格式选项:</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let pi = 3.141592;</span></span>
<span class="line"><span>let formatted = format!(&quot;{:.2}&quot;, pi); // 保留两位小数</span></span>
<span class="line"><span>// 输出: &quot;3.14&quot;</span></span>
<span class="line"><span>println!(&quot;{}&quot;, formatted);</span></span></code></pre></div><p>format! 宏非常灵活，允许各种格式选项和参数类型。这使得它在需要动态构建字符串时非常有用。</p><p>\\</p>`,71)]))}const g=n(t,[["render",l]]);export{h as __pageData,g as default};
