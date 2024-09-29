import{_ as n,c as a,a0 as p,o as t}from"./chunks/framework.DOEc3Y8C.js";const g=JSON.parse('{"title":"ownership","description":"","frontmatter":{},"headers":[],"relativePath":"language/rust/6.ownership.md","filePath":"language/rust/6.ownership.md"}'),l={name:"language/rust/6.ownership.md"};function e(i,s,o,c,r,d){return t(),a("div",null,s[0]||(s[0]=[p(`<h1 id="ownership" tabindex="-1">ownership <a class="header-anchor" href="#ownership" aria-label="Permalink to &quot;ownership&quot;">​</a></h1><p><strong>所有权</strong></p><p>是rust的特色，由于所有权是一个新概念，因此需要花费一些时间来掌握它。</p><p>所有的程序都必须和计算机内存打交道，如何从内存中申请空间来存放程序的运行内容，如何在不需要的时候释放这些空间，成了重中之重，也是所有编程语言设计的难点之一。在计算机语言不断演变过程中，出现了三种流派：</p><ul><li><p><strong>垃圾回收机制(GC)</strong>，在程序运行时不断寻找不再使用的内存，典型代表：Java、Go</p></li><li><p><strong>手动管理内存的分配和释放</strong>, 在程序中，通过函数调用的方式来申请和释放内存，典型代表：C++</p></li><li><p><strong>通过所有权来管理内存</strong>，编译器在编译时会根据一系列规则进行检查</p><p>这里需要了解下数据结构中的栈和堆</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>rust程序设计中文版本解释如下：</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>栈和堆都是代码在运行时可供使用的内存，但是它们的结构不同。栈以放入值的顺序存储值并以相反顺序取出值。这也被称作 后进先出（last in, first out）。想象一下一叠盘子：当增加更多盘子时，把它们放在盘子堆的顶部，当需要盘子时，也从顶部拿走。不能从中间也不能从底部增加或拿走盘子！增加数据叫做 进栈（pushing onto the stack），而移出数据叫做 出栈（popping off the stack）。</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>栈中的所有数据都必须占用已知且固定的大小。在编译时大小未知或大小可能变化的数据，要改为存储在堆上。堆是缺乏组织的：当向堆放入数据时，你要请求一定大小的空间。内存分配器（memory allocator）在堆的某处找到一块足够大的空位，把它标记为已使用，并返回一个表示该位置地址的 指针（pointer）。这个过程称作 在堆上分配内存（allocating on the heap），有时简称为 “分配”（allocating）。将数据推入栈中并不被认为是分配。因为指针的大小是已知并且固定的，你可以将指针存储在栈上，不过当需要实际数据时，必须访问指针。</span></span>
<span class="line"><span>想象一下去餐馆就座吃饭。当进入时，你说明有几个人，餐馆员工会找到一个够大的空桌子并领你们过去。如果有人来迟了，他们也可以通过询问来找到你们坐在哪。</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>入栈比在堆上分配内存要快，因为（入栈时）分配器无需为存储新数据去搜索内存空间；其位置总是在栈顶。相比之下，在堆上分配内存则需要更多的工作，这是因为分配器必须首先找到一块足够存放数据的内存空间，并接着做一些记录为下一次分配做准备。</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>访问堆上的数据比访问栈上的数据慢，因为必须通过指针来访问。现代处理器在内存中跳转越少就越快（缓存）。继续类比，假设有一个服务员在餐厅里处理多个桌子的点菜。在一个桌子报完所有菜后再移动到下一个桌子是最有效率的。从桌子 A 听一个菜，接着桌子 B 听一个菜，然后再桌子 A，然后再桌子 B 这样的流程会更加缓慢。出于同样原因，处理器在处理的数据彼此较近的时候（比如在栈上）比较远的时候（比如可能在堆上）能更好的工作。在堆上分配大量的空间也可能消耗时间。</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>当你的代码调用一个函数时，传递给函数的值（包括可能指向堆上数据的指针）和函数的局部变量被压入栈中。当函数结束时，这些值被移出栈。</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>跟踪哪部分代码正在使用堆上的哪些数据，最大限度地减少堆上的重复数据量，以及清理堆上不再使用的数据确保不会耗尽空间，这些问题正是所有权系统要处理的。一旦理解了所有权，你就不需要经常考虑栈和堆了，不过明白了所有权的存在就是为了管理堆数据，能够帮助解释为什么所有权要以这种方式工作。</span></span>
<span class="line"><span>​</span></span></code></pre></div></li></ul><p>rust的所有权规则</p><ul><li><p>Rust 中的每一个值都有一个被称为其 <strong>所有者</strong>（<em>owner</em>）的变量。</p></li><li><p>值在任一时刻有且只有一个所有者。</p></li><li><p>当所有者（变量）离开作用域，这个值将被丢弃。</p><p>这些规则的<strong>作用</strong>和<strong>好处</strong>就有了以下几点：</p><ol><li><strong>防止双重释放</strong>：由于每个值只有一个所有者，当该值离开作用域时，它只会被删除一次。这避免了双重释放内存的风险。</li><li><strong>有效的内存管理</strong>：不需要运行时垃圾收集器。当数据不再使用时，Rust确保它立即被清理。</li><li><strong>数据竞争安全</strong>：由于同一时间只能有一个可变引用或任意数量的不可变引用，Rust在编译时防止了数据竞争。</li><li><strong>清晰的代码意图</strong>：所有权和借用规则迫使开发者明确地考虑如何使用数据，这导致了更明确和更容易理解的代码意图。</li><li><strong>无需手动内存管理</strong>：尽管Rust提供了手动内存管理的工具，但大多数情况下，你不需要担心分配和释放内存，Rust为你处理了这些。</li><li><strong>并发友好</strong>：所有权规则使得并发编程更为安全，因为它在编译时检查潜在的数据竞争和竞态条件。</li></ol></li></ul><p><strong>移动</strong></p><p>在Rust中，当你将一个变量赋值给另一个变量时，原始数据实际上是被移动的，不是复制的（除非这个类型实现了<code>Copy</code> trait）。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let s1 = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>let s2 = s1;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// 此时s1已经不再有效，因为String的数据被移动到了s2</span></span>
<span class="line"><span>// println!(&quot;{}&quot;, s1); // 这会导致编译错误</span></span></code></pre></div><p><strong>绑定</strong></p><p>在Rust中，&quot;绑定&quot;是一个核心概念，涉及变量与其关联的数据。当你将一个值赋给一个变量时，这个变量实际上是绑定到那个值的。这意味着变量与其值之间建立了一个关联。</p><p>绑定的种类：</p><ol><li><p><strong>模式绑定</strong>: 当你在模式匹配、结构体解构或迭代中使用变量时，你实际上是在创建绑定。例如，在模式匹配中：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>match value {</span></span>
<span class="line"><span>    Some(x) =&gt; println!(&quot;{}&quot;, x),  // 这里x绑定到Some内部的值</span></span>
<span class="line"><span>    None =&gt; (),</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在上述代码中，<code>x</code>绑定到<code>Some</code>中的值。</p></li><li><p><strong>函数参数绑定</strong>: 当你传递一个值到函数时，函数的参数会绑定到那个值。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fn print_value(x: i32) {</span></span>
<span class="line"><span>    println!(&quot;{}&quot;, x);  // 这里x绑定到传递给函数的值</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>let绑定</strong>: 这是最常见的绑定形式。当你使用<code>let</code>关键字声明一个变量时，你创建了一个新的绑定。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let x = 5;  // x绑定到值5</span></span></code></pre></div></li></ol><p>可变与不可变绑定：</p><ul><li><p><strong>不可变绑定</strong>: 默认情况下，绑定是不可变的，这意味着你不能更改其绑定的值。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let x = 5;</span></span>
<span class="line"><span>// x = 6;  // 这会导致编译错误，因为x是不可变的</span></span></code></pre></div></li><li><p><strong>可变绑定</strong>: 使用<code>mut</code>关键字，你可以创建一个可变绑定，这意味着你可以更改其绑定的值。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let mut y = 5;</span></span>
<span class="line"><span>y = 6;  // 这是允许的，因为y是可变的</span></span></code></pre></div></li></ul><p><strong>克隆</strong></p><p>如果你确实需要复制某些数据，而不是移动，可以使用<code>.clone()</code>方法（如果该类型支持克隆）。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let s1 = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>let s2 = s1.clone();</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>println!(&quot;s1 = {}, s2 = {}&quot;, s1, s2); // 这是有效的</span></span></code></pre></div><p><strong>函数和所有权</strong></p><p>当你将一个值传递给函数时，与赋值给变量的情况类似，该值也会被移动或复制。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fn takes_ownership(some_string: String) {</span></span>
<span class="line"><span>    println!(&quot;{}&quot;, some_string);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>takes_ownership(s);</span></span>
<span class="line"><span>// s在这里不再有效，因为它的所有权已被移动到函数中</span></span></code></pre></div><p><strong>引用和借用</strong></p><p>为了允许访问数据但不取得其所有权，Rust引入了引用和借用的概念。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fn calculate_length(s: &amp;String) -&gt; usize {</span></span>
<span class="line"><span>    s.len()</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>let len = calculate_length(&amp;s);</span></span>
<span class="line"><span>// s在这里仍然有效，因为我们只是传递了一个引用</span></span></code></pre></div><p>需要注意的是，Rust默认不允许两个或多个可变引用存在于同一作用域，也不允许同时存在可变引用和不可变引用，以确保数据的安全性。</p><p><strong>在借用之中当一个变量有引用存在时，不能移交它所绑定的数据的所有权</strong></p><p>解释：它确保了内存安全和数据一致性。</p><p>首先，让我们明确什么是&quot;有引用存在&quot;。当我们说一个变量被借用，实际上是说这个变量的数据被其他变量引用了。</p><p>现在，考虑以下示例：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let mut s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>let r = &amp;s;  // s现在被r引用或借用</span></span></code></pre></div><p>在上面的代码中，我们创建了一个<code>String</code>类型的变量<code>s</code>，然后我们创建了一个引用<code>r</code>，它引用（或借用）<code>s</code>的数据。</p><p>此时，根据Rust的规则，你不能改变<code>s</code>的值或移交它的所有权，因为它被<code>r</code>引用了。这样做是为了避免数据竞争、悬挂引用和其他可能的错误。</p><p>例如，以下操作都是不允许的：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let mut s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>let r = &amp;s;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// 试图修改s的值</span></span>
<span class="line"><span>s.push_str(&quot;, world&quot;); // 这会导致编译错误，因为s正在被r引用</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// 试图移交s的所有权</span></span>
<span class="line"><span>let s2 = s; // 这也会导致编译错误，因为s的所有权不能在被r引用时被移交</span></span></code></pre></div><p>简而言之，当一个变量被其他变量引用（或借用）时，你不能更改它的值或移交它的所有权。这是为了确保所有引用始终指向有效和一致的数据，从而避免潜在的内存安全问题。</p><p>在这里面还要区分<code>不可变引用</code>和<code>可变引用</code>。</p><ol><li><p><strong>对于不可变引用 (<code>&amp;T</code>)</strong>:</p><ul><li>你<strong>不能</strong>通过任何引用来修改该值。</li><li>你<strong>不能</strong>更改原始变量的值。</li><li>你<strong>不能</strong>移交原始变量的所有权。</li></ul><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>let r = &amp;s; // 不可变引用</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// 下面的所有操作都是不允许的</span></span>
<span class="line"><span>// s.push_str(&quot;, world&quot;); // 试图修改s的值</span></span>
<span class="line"><span>// let s2 = s; // 试图移交s的所有权</span></span></code></pre></div></li><li><p><strong>对于可变引用 (<code>&amp;mut T</code>)</strong>:</p><ul><li>你<strong>可以</strong>通过这个可变引用来修改该值。</li><li>但在存在活跃的可变引用时，你<strong>不能</strong>更改原始变量的值。</li><li>你<strong>不能</strong>移交原始变量的所有权。</li></ul><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let mut s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>let r = &amp;mut s; // 可变引用</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>r.push_str(&quot;, world&quot;); // 通过r修改s的值是允许的</span></span>
<span class="line"><span>// s.push_str(&quot;!&quot;); // 这是不允许的，因为s有一个活跃的可变引用</span></span>
<span class="line"><span>// let s2 = s; // 这也是不允许的，因为s有一个活跃的可变引用</span></span></code></pre></div></li></ol><p><strong>下面是我和chatgpt的一些对话来解决我的困惑</strong></p><p>我：我现在有2个问题，第一，如果在引用期间对可变变量的原始值进行更改，那么引用会随原始值进行更改吗？？</p><p>第二，如果不小心将某个变量设置为不可变，但在引用时需要修改它的值，可以重新定义它吗？</p><p>chatgpt：</p><ul><li><strong>如果在引用期间对可变变量的原始值进行更改，那么引用会随原始值进行更改吗？</strong></li></ul><p>答案是：不，你不能在拥有活跃引用的同时修改原始变量的值，无论是不可变引用还是可变引用。如果你尝试这样做，Rust编译器会给出错误。引用的主要目的是确保数据安全性，防止意外的数据修改和数据竞争。因此，当一个活跃的引用存在时，你不能修改其指向的数据。</p><p>示例：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let mut s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>let r = &amp;mut s; // 创建一个可变引用</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// s.push_str(&quot;, world&quot;); // 这会导致编译错误，因为s有一个活跃的引用</span></span></code></pre></div><ul><li><strong>如果不小心将某个变量设置为不可变，但在引用时需要修改它的值，可以重新定义它吗？</strong></li></ul><p>答案是：是的，你可以为该变量重新分配一个新的值，这会隐式地使之前的所有引用都失效。但是，如果你的意思是你想在已经创建了引用的情况下重新定义或修改这个变量，那么你不能这样做，因为这会违反Rust的所有权和引用规则。</p><p>但如果没有任何活跃的引用，你可以轻松地给一个不可变变量重新赋值：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>let mut s = s; // 重新定义s为可变</span></span>
<span class="line"><span>s.push_str(&quot;, world&quot;); // 现在这是允许的，因为s是可变的</span></span></code></pre></div><p>在上述代码中，我们首先定义了一个不可变的<code>String</code>变量<code>s</code>。然后，我们重新定义了一个可变的<code>s</code>变量，并为其分配了原始<code>s</code>的值。这是完全有效的，因为我们没有违反任何规则或持有任何活跃的引用。</p><p>我：这里的活跃引用在怎么理解</p><p>chatgpt：&quot;活跃引用&quot;是指在当前作用域或上下文中有效并可以被使用的引用。当一个引用被创建并指向某个值时，这个引用就是活跃的，直到它超出其作用域或被显式地使其失效。</p><p>让我们通过几个例子来理解这个概念：</p><ul><li><strong>基本作用域内的活跃引用</strong>：</li></ul><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fn main() {</span></span>
<span class="line"><span>    let mut s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>    {</span></span>
<span class="line"><span>        let r = &amp;mut s; // r是一个活跃的可变引用</span></span>
<span class="line"><span>        // 在这个作用域中，r是活跃的</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    // 当我们离开上面的作用域时，r不再是活跃的</span></span>
<span class="line"><span>    s.push_str(&quot;, world&quot;); // 这是允许的，因为现在没有活跃的引用</span></span>
<span class="line"><span>}</span></span></code></pre></div><ul><li><strong>在函数中的引用</strong>：</li></ul><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fn main() {</span></span>
<span class="line"><span>    let mut s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>    let r = &amp;mut s;  // r是一个活跃的可变引用</span></span>
<span class="line"><span>    modify(r);</span></span>
<span class="line"><span>    // 在函数modify调用结束后，r仍然是活跃的，因为它在main函数的作用域中定义</span></span>
<span class="line"><span>    // s.push_str(&quot;, world&quot;); // 这会导致编译错误，因为r仍然是活跃的</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>fn modify(some_string: &amp;mut String) {</span></span>
<span class="line"><span>    some_string.push_str(&quot;, world&quot;);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在上述代码中，尽管<code>modify</code>函数已经返回，但在<code>main</code>函数中，<code>r</code>仍然是一个活跃的引用，因此我们不能直接修改<code>s</code>。</p><ul><li><strong>使引用失效</strong>：</li></ul><p>在某些情况下，通过重新分配值，你可以显式地使引用失效。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let mut s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>let r1 = &amp;s;  // r1是一个活跃的不可变引用</span></span>
<span class="line"><span>let r2 = &amp;s;  // r2也是一个活跃的不可变引用</span></span>
<span class="line"><span>s = String::from(&quot;world&quot;);  // s被重新赋值，这使得r1和r2都失效</span></span></code></pre></div><p>总之，&quot;活跃引用&quot;是指在其作用域内有效并可以被使用的引用。当引用超出其作用域或被显式地使其失效时，它就不再是活跃的。Rust编译器使用这些规则来确保内存安全和数据一致性。</p><p>我：这里的显式和隐式怎么理解</p><p>chatgpt：</p><p><strong>显式 (Explicit)</strong></p><p>当我们说某件事是&quot;显式&quot;的，意思是它在代码中被明确地指定或声明。你可以直接从代码中看到并理解这个操作或行为。它需要程序员直接介入并指明要发生的事情。</p><p><strong>示例</strong>:</p><ol><li><p><strong>显式类型注解</strong>: 在Rust中，虽然大多数时候编译器可以推断变量的类型，但你也可以显式地指定它。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span> let x: i32 = 5;  // i32类型被显式地指定</span></span></code></pre></div></li><li><p><strong>显式地调用函数</strong>: 当你明确地调用函数或方法。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let y = String::from(&quot;hello&quot;);  // 显式地调用from函数</span></span></code></pre></div></li></ol><p><strong>隐式 (Implicit)</strong></p><p>当我们说某件事是&quot;隐式&quot;的，意思是它在代码中并没有明确地表示出来，但它仍然会发生。这通常是由编程语言的规则、编译器的推断或某些默认行为导致的。</p><p><strong>示例</strong>:</p><ol><li><p><strong>隐式类型推断</strong>: 在Rust中，编译器经常会根据上下文推断变量的类型。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let x = 5;  // x的类型隐式地被推断为i32</span></span></code></pre></div></li><li><p><strong>隐式实现trait</strong>: 在某些情况下，Rust的类型会隐式地实现某些trait，如<code>Copy</code> trait，这取决于其内容。</p></li><li><p><strong>隐式转换</strong>: 在某些编程语言中，不同的数据类型之间可能会有隐式转换。例如，在某些语言中，浮点数可能会隐式地转换为整数。</p></li></ol><p>我：活跃引用超出作用域我是不是可以理解为生命周期结束了</p><p>chatgpt回答：是的，你的理解是正确的。当一个引用超出其作用域时，可以说它的&quot;生命周期&quot;已经结束。</p><p>在Rust中，每个引用都有一个称为&quot;生命周期&quot;的概念，这是一个编译时的概念，表示引用在何时有效或&quot;活跃&quot;。生命周期用于预防悬挂引用或无效引用，确保所有的引用都是安全的。</p><p>当你在代码中创建一个引用时，它的生命周期从被创建的地方开始，直到其作用域结束。一旦超出了这个作用域，引用就不再有效，因此它的生命周期就结束了。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>{</span></span>
<span class="line"><span>    let s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>    let r = &amp;s;  // r的生命周期从这里开始</span></span>
<span class="line"><span>    // 在这个范围内，r是有效的</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>// 一旦我们离开上面的块，r的生命周期就结束了，r不再有效</span></span></code></pre></div><p>在某些复杂的情况下，Rust可能需要显式的生命周期注解来理解引用的有效期，但在很多常见的情况下，Rust可以自动推断出正确的生命周期。</p><p>\\</p>`,80)]))}const h=n(l,[["render",e]]);export{g as __pageData,h as default};