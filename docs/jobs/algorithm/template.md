## 技巧
取消同步输入输出流：ios::sync_with_stdio(0);cin.tie(0);cout.tie(0);
$exit(0)$ 用于子函数结束程序

## 区间dp
```cpp
for(int k=1;k<n;k++)
for(int l=1,r=l+1;r<=n;l++,r++)
for(int i=l;i<r;i++)
{
dp[l][r]=min||max(dp[l][i]+dp[i+1][r]+C,dp[l][r]);
}
```

## 并查集
```cpp
vector<int>father=vector<int>(size);
int sets;
void build(int m){
	for(int i=0;i<m;i++){
		father[i]=i;
	}
	sets=m;
}
int find(int i){
	if(i!=father[i]){
		father[i]=find(father[i]);
	}
	return father[i];
}
void uni(int x,int y){
	int fx=find(x);
	int fy=find(y);
	if(fx!=fy){
		father(fx)=fy;
		sets--;
	}
}
```

## Dijkstra
```cpp
#include<iostream>
#include<cstring>
#include<queue>
#include<algorithm>
#define mm 20000001
#define inf 0x7fffffff
using namespace std;
int n,m,s,cnt;
int h[2000001],dis[mm];
bool visit[mm];
struct Edge{
    int to,w,next;
}edge[2000001];
struct node
{
    int dis;
    int pos;
    bool operator <( const node &x )const
    {
        return x.dis < dis;
    }
};
priority_queue<node>q;
void init ()
{
    memset(visit,0,sizeof(visit));
    memset(dis,0x3f,sizeof(dis));
    memset(h,-1,sizeof(h));
}
void add(int u,int v,int w)
{
    edge[cnt].to=v,edge[cnt].w=w,edge[cnt].next=h[u],h[u]=cnt++;
}
void pintf()
{
    for(int i=1;i<=n;i++)
    {
        cout<<dis[i]<<' ';
    }
}
void Dijkstra()
{
    while(!q.empty())
    {
    	node temp=q.top();
    	q.pop();
        int x=temp.pos,d=temp.dis;
        if(!visit[x])
        {
        	visit[x]=1;
        for(int i=h[x];i!=-1;i=edge[i].next)
        {
            int t=edge[i].to;
            if(!visit[t]&&dis[t]>dis[x]+edge[i].w)
            dis[t]=dis[x]+edge[i].w,q.push((node){dis[t],t});
        }
        }
    }
    return ;
}
int main ()
{
    cin>>n>>m>>s;
    init();
    for(int i=1;i<=m;i++)
    {
        int a,b,c;
        cin>>a>>b>>c;
        add(a,b,c);
    }
    dis[s]=0;
    q.push((node){0,s});
    Dijkstra();
    pintf();
    return 0;
}
```

## 树状数组
```cpp
int lowbit(int x)
{
    return x & (-x);
}
void add(int v, int x)
{
    for (int i = v; i <= n; i += lowbit(i))
    {
        c[i] += x;
    }
    return;
}
int query(int x)
{
    int sum = 0;
    for (int i = x; i >= 1; i -= lowbit(i))
    {
        sum += c[i];
    }
    return sum;
}

```

## LCS 和 LIS 问题
LIS ：最长单调上升子序列。
LCS：最长公共子序列。

### LIS
考虑 $O(n)$ 单调上升子序列，
易得出一下代码，
```cpp
 for(int i=1;i<=n;i++)
     {
        f[i]=1;
       for(int j=1;j<i;j++)
       {
        if(a[i]>=a[j])
         f[i]=max(f[i],f[j]+1);
       }
     }
```
可优化为 $O(nlogn)$，考虑单调栈，每次往栈内压入一个 $x$ 保证 $f_{j''}<x<f_{j'},j''<j'$ ，所以原来栈内元素中把 $f_{j'}$ 弹出用 $x$ 替换掉，即 $f_{j'}=x$ 。保证栈内单调，且优先每个最大子段数保证递增的末尾数值尽可能的小。
而如何求出来 $j'$ ，一种是模拟的求，$O(n)$，由于栈内保证单调，所以也可以二分求出来 $O(logn)$ 。
当然一般会用 $k=lower\_bound(f+1,f+t+1,x)-f$ 来求。
代码展示
```cpp
for (int i = n; i >= 1; i--)
    {
        int k = upper_bound(f + 1, f + ans + 1, a[i]) - f;
        f[k] = a[i];
        ans = max(ans, k);
    }
```

### LCS
首先考虑朴素的公共子序列问题
典型的区间 $dp$ 问题
```cpp
for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++)
        {
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            if (a1[i] == a2[j])
                dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1);
            // 因为更新，所以++；
        }
```
回文也可以简单的理解为颠转顺序来求正反串的最大公共子序列。
考虑找到最大公共子序列，用强制定义顺序，来使 LCS转换成LIS问题。

对于任意两段数字内容序列，首先将其离散化，对任意序列强行定义顺序，在将另一序列用定义的顺序替换即可。
$for$ $example$ 
3 2 1 4 5
4 2 3 5 1
将第二个队列用 1 2 3 4 5强行定义递增顺序，第一队列按照第二队列对应的方式一一映照即可。
得出3 2 5 1 4 
然后再在3 2 5 1 4中找LIS就是该两队列中的LCS。
映射代码：
```cpp
for (int i = 1; i <= n; i++)
    {
        int x;
        cin>> x;
        v[x] = i;
    }
```

## 位运算
#### 求某个数二进制的第i位：$x\&(1<<i)$

$(1<<i)$ 表示 $2^i$，$(n<<i)$ 表示 $n*2^i$，$(i>>k)\&1$  表示二进制数 $i$ 的从右往左数第 $k$ 位是否为 $1$

二进制中 -x=~x+1；
$lowbit(x)$ 表示返回 $x$ 最后一位为 $1$ 的数

- $k$ 个相同的数异或和，当$k$ 为奇数，结果为这个数本身，否则为 $0$ 。
- 任何数与 $0$ 异或是这个数本身。
```cpp
int lowbit(int x)
{
   return x&((~x)+1);
}
```

## 快速幂
```cpp
#include <bits/stdc++.h>
using namespace std;
#define int long long
const int N = 1001;
int a, b, p;
int n, m, ans;
int t;
void qmi()
{
    int res = 1 % p;
    while (b)
    {
        if (b & 1)
            res = res * a % p;
        a = a * a % p;
        b /= 2;
    }
    cout << res << "\n";
}
signed main()
{
    cin >> t;
    while (t--)
    {
        cin >> a >> b >> p;
        qmi();
    }
    return 0;
}

```

## 进制问题

X 进制：

平时我们所说的10进制数是怎么得出来的呢？
比如10进制数 123: 它是由百位上的1 * 10 * 10 加上 十位上的 2 * 10 加上 个位上的 3 得出来的


关于x进制转10进制：
比如题目中给的：11进制（10）、5进制（4）、2进制（0）
                对于i位上的数字 $num[i]$，转换为十进制就是 $num[i]$ *低于i位所有位的进制
                
                就是10*5*2+4*2+0=108
        再比如：11进制（1）、5进制（2）、2进制（0）       
         
                就是1*5*2+2*2+0=14

## 组合数学
{
插板法：[解释](https://www.zhihu.com/question/422265718/answer/1937189993)
公式：$C_{n-1}^ {k-1}$
```cpp
void build()
{
       c[0][0] = c[1][0] = c[1][1] = 1;
       for (int i = 2; i <= 2000; i++)
       {
              c[i][0] = 1;
              c[i][i] = 1;
              for (int j = 1; j <= i; j++)
              {
                     c[i][j] = (c[i - 1][j - 1] + c[i - 1][j]) % k;
                     f[i][j] = f[i - 1][j] + f[i][j - 1] - f[i - 1][j - 1];
                     if (c[i][j] == 0)
                            f[i][j]++;
                     f[i][i + 1] = f[i][i];
              }
       }
}
```
}
## 进制转换
```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 101001;

string a, b;
map<char, int> q;
int tra10(int n, string s)
{
    int ns = s.size();
    int k = 1;
    int res = 0;
    for (int i = ns - 1; i >= 0; i--)
    {
        if (s[i] >= 'A' && s[i] <= 'F')
            res += (q[s[i]] * k);
        else
            res += (s[i] - '0') * k;
        k *= n;
    }
    return res;
}
void output(stack<int> s)
{
    while (!s.empty())
    {
        if (s.top() >= 10)
        {
            cout << (char)(s.top() + 'A' - 10);
        }
        else
            cout << s.top();
        s.pop();
    }
}
void work(int n, int r)
{
    stack<int> s;
    while (n)
    {
        s.push(n % r);
        n /= r;
    }
    output(s);
}
int main()
{
    int n, m, ans;
    cin >> n >> a >> m;
    q['A'] = 10;
    q['B'] = 11;
    q['C'] = 12;
    q['D'] = 13;
    q['E'] = 14;
    q['F'] = 15;
    int n_10 = tra10(n, a);
    // cout << n_10;
    work(n_10, m);
    return 0;
}
```

## 可持久化线段树（主席树）
```cpp
// 对于建树的存数都是优先存到左区间里
#include <bits/stdc++.h>
#define mm 300000
using namespace std;
// 定义区
struct nihao
{
    int l, r, sum, nl, nr, Sum; // 历史的树根节点，左子树，右子树(都是指的节点)//某个区间有几个数
} t[mm << 5];
int a[mm], b[mm]; // 原数组，离散数组
int nodee;        // 节点
int inst;         // 修改插入的点
int g;            // 离散的右边界
int n, m, k;
int anss;
int rt[mm << 2];
// 函数区
void make(int &x, int l, int r) // 建树
{
    x = ++nodee;
    if (l == r)
        return; // 叶节点
    int mid = (l + r) >> 1;
    make(t[x].l, l, mid);
    make(t[x].r, mid + 1, r);
    return;
}

void modify(int &xi, int x, int l, int r, int v)
{
    xi = ++nodee; // 建立新节点
    t[xi] = t[x];
    t[xi].sum++;
    t[xi].Sum += v; // 传递树节点数据
    if (l == r)
        return; // 已经到叶节点，不用再建树了
    int mid = (l + r) >> 1;
    if (inst <= mid)
        modify(t[xi].l, t[x].l, l, mid, v);
    else
        modify(t[xi].r, t[x].r, mid + 1, r, v);
    return;
}

int query(int use, int now, int l, int r, int k)
{
    int ans;
    int mid = (l + r) >> 1;
    int h = t[t[now].l].sum - t[t[use].l].sum; // h是权值线段树操作
    if (l == r)
        return l;
    if (k <= h)
        ans = query(t[use].l, t[now].l, l, mid, k); // 递归的都是节点
    else
        ans = query(t[use].r, t[now].r, mid + 1, r, k - h); // 注意要减去左区间的数
    return ans;
}
signed /*int*/ main()
{
    int l, r;
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
        cin >> a[i], b[i] = a[i];
    sort(b + 1, b + n + 1);
    g = unique(b + 1, b + n + 1) - b - 1;
    make(rt[0], 1, g);
    for (int i = 1; i <= n; i++) // 全部建树
    {
        inst = lower_bound(b + 1, b + g + 1, a[i]) - b; // 找到a[i]对应的离散化下标
        modify(rt[i], rt[i - 1], 1, g, a[i]);           // modify(rt[i],rt[i-1],1,inf,a[i]);
    }
    while (m--)
    {
        cin >> l >> r;
        // k = re;

        k = (r - l) / 2 + 1;
        //  cout << k << " ";
        anss = query(rt[l - 1], rt[r], 1, g, k);
        cout << b[anss] << "\n";
    }
    return 0;
}
```

## 单调队列
**单调队列维护前缀和的时候，由于用到 $s[i]-s[0]$ 所以在队列开始的时候就要存入 $q[h]=0$ ，那么必然要把初始值设置为 $h=0,t=1$ **

 更新尾指针的时候，比较的下标是 $t-1$

**双指针近乎可以优化掉所有的区间内的大小问题（某种数据需要具有单调性）**

很多的时候往往要用到初始的长区间序列，所以定义的时候就要保证h=0,t=1;(一般用于前缀和问题，防止初始元素被剪掉)

考虑在 $dp$ 转移的时候， $f_i=max(f_j)+c$ ， $j$ 在某个可计算大小的区间内，在 $dp$ 随着 $i$ 进行转移的过程中只需要维护 $j$ 所在的区间的最大值即可。

即若 $dp$ 转移的过程中，存在从某一区间进行的转移，均可考虑使用单调队列进行优化。

一般的优化考虑的是使用双端队列来优化单调队列，个人认为也可以考虑使用优先队列（这个用来候补，暂时还想不到如何存储数据） 


在动态区间维护区间最值也是可以的，可以添加一些性质（平均数）

[P1419 寻找段落](https://www.luogu.com.cn/problem/P1419)

前缀和单调队列优化 [AcWing 135. 最大子序和](https://www.acwing.com/file_system/file/content/whole/index/content/10747030/) [AcWing 1087. 修剪草坪 ](https://www.acwing.com/file_system/file/content/whole/index/content/10747187/)

甚至可以考虑使用二维的。
### 维护过程

考虑维护区间最值时，使用单调队列进行优化，考虑每往队列里面添加一个元素 $x$，那么之后在队列内保留的元素 $y$ 必然满足 $y>x$。队列内部满足单调性，过期元素自动剔除，队首满足最大值和过期最小值，队末满足队列内最小值和过期最大值。

队列中只需记录元素下标即可，队列中的元素大小递减，过期值递增。

```cpp
  const int N = 11111111;
  int q[N],d[N];// 数组原值，队列数组
  int h=0,t=0,k; //头指针，尾指针，区间大小

  for(int i=1;i<=n;i++)
  {
    while(h<t&&q[d[t-1]]<=q[i])t--;
    d[t++]=i;
    while(d[t-1]-d[h]>k)h++;  
  }
```

考虑另一种单调队列的写法
```cpp
int h = 0, t = 0;
int q[100100], pos[100012];//此时的q数组相当于原来的a数组
int kk = N;//定义一个可计算的区间
for(int i=1;i<=n;i++)
{
  while(h<t&&pos[h]>kk)h++
  int x;//一个被计算的需要录入的值
  while(h<t&&q[t-1]<=x)t--;
  pos[t]=i;
  q[t++]=x;
}
```


## 线段树

来历已久，久仰大名了。
线段树是解决区间问题，区间加，区间减，区间最值，区间覆盖，凡是可以将区间合并，并快速的寻找任意区间的任意值的话，都可用线段树解决。
所以线段树>st表~树状数组。
线段树分为的步骤：build（建树），push_up（上传结点信息），push_down（下传结点信息），make_tag（区间信息维护），query（区间查询），up_date（结点维护），find（判断建树结果）。
线段树乘法要先乘后加。

**up_date中有push_down和push_up,query中有push_down**

**其实线段树的优点就是快速的维护区间信息的东西**

可以维护，公倍数，公因数，最大值，最小值，只要满足区间就可以使用线段树

### 结构部分


- 树的结构：date，l，r，lz（懒标记）。
```cpp
struct nihao
{
    int l, r, date, lz, mx, mn, lx, ln;
} t[2813102];
```

### 可以开一个答案查询函数
```cpp
const int inf = 231231231;
nihao ans;
void init()
{
    ans.mx = -inf;
    ans.mn = inf;
    ans.date = 0;
}
void ans_query(int num)
{
    ans.mx = max(ans.mx, t[num].mx);
    ans.mn = min(ans.mn, t[num].mn);
    ans.date += t[num].date;
}
```

- build（建树
```cpp
void build(int l, int r, int num)
{
    t[num].l = l, t[num].r = r;
    if (l == r)
    {
        t[num].date = p[l];
        t[num].mx = p[l];
        t[num].mn = p[l];
        t[num].lx = -inf;
        t[num].ln = inf;
        return;
    }
    int mid = (l + r) >> 1;
    build(l, mid, num << 1);
    build(mid + 1, r, num << 1 | 1);
    push_up(num);
}
```
- push_up（上传结点信息）
```cpp
void push_up(int num)
{
    t[num].date = t[num << 1].date + t[num << 1 | 1].date;
    t[num].mx = max(t[num << 1].mx, t[num << 1 | 1].mx);
    t[num].mn = min(t[num << 1].mn, t[num << 1 | 1].mn);
    t[num].lx = -inf;
    t[num].ln = inf;
}
```
- push_down（下传结点信息）
```cpp
void push_down(int num)
{
    int mid = (t[num].l + t[num].r) >> 1;
    make_tag(num << 1, t[num].lz, t[num].lx, t[num].ln);
    make_tag(num << 1 | 1, t[num].lz, t[num].lx, t[num].ln);
    t[num].lz = 0;
    t[num].lx = -inf;
    t[num].ln = inf;
}
```
- make_tag（区间信息维护）
```cpp
void make_tag(int num, int k, int kk, int kkk) //
{
    t[num].mx = max(t[num].mx, kk);
    t[num].mn = min(t[num].mn, kkk);
    t[num].date += k * (t[num].r - t[num].l + 1);
    t[num].lz += k;
    t[num].lx = max(t[num].lx, kk);
    t[num].ln = min(t[num].ln, kkk);
}
```
- query（区间查询）
```cpp
void query(int l,int r,int num)//查询区间的l,r不会变，只有子树的区间会变
{
    if (in(t[num].l, t[num].r, l, r))
    {
        ans_query(num);
        return;
    }
    if (out(t[num].l, t[num].r, l, r))
    {
        return;
    }
    int mid = (t[num].l + t[num].r) >> 1;
    push_down(num);
    if (l <= mid)
        query(l, r, num << 1);
    if (r > mid)
        query(l, r, num << 1 | 1);
    return;
}
```
- up_date（结点维护）
```cpp
void up_date(int l,int r,int k,int num)
{
    if (out(l, r, t[num].l, t[num].r))
        return;
    if (in(t[num].l, t[num].r, l, r))
    {
        make_tag(num, k, k, k);
        return;
    }
    push_down(num);
    up_date(l, r, k, num << 1);
    up_date(l, r, k, (num << 1) + 1);
    push_up(num);  
}
```
最后加上一个 $find$ 查找函数
- find （判断建树结果）
```cpp
void find(int l, int r, int num)
{
    int mid = (l + r) >> 1;
    if (t[num].l == 0 || t[num].r == 0)
        return;
    cout << t[num].l << " " << t[num].r << " " << t[num].date << " " << t[num].mx << " " << t[num].mn << "\n";
    if (l == r)
    {
        return;
    }
    find(num << 1, l, mid);
    find(num << 1 | 1, mid + 1, r);
    return;
}
```

### 源代码

```cpp
#include <bits/stdc++.h>
using namespace std;

#define int long long

const int inf = 231231231;
struct nihao
{
    int l, r, date, lz, mx, mn, lx, ln;
} t[2813102];
int p[213123112];
int n;
// 可以开一个答案查询函数
nihao ans;
void init()
{
    ans.mx = -inf;
    ans.mn = inf;
    ans.date = 0;
}
void push_up(int num)
{
    t[num].date = t[num << 1].date + t[num << 1 | 1].date;

    t[num].mx = max(t[num << 1].mx, t[num << 1 | 1].mx);
    t[num].mn = min(t[num << 1].mn, t[num << 1 | 1].mn);
    // cout << t[num << 1].l << " " << t[num << 1 | 1].r << " " << t[num << 1].mx << " " << t[num << 1].mn << " " << t[num].mx << " " << t[num].mn << " "
    //	 << "\n";
    t[num].lx = -inf;
    t[num].ln = inf;
}
void build(int l, int r, int num)
{
    t[num].l = l, t[num].r = r;
    if (l == r)
    {
        t[num].date = p[l];
        t[num].mx = p[l];
        t[num].mn = p[l];
        t[num].lx = -inf;
        t[num].ln = inf;
        return;
    }
    int mid = (l + r) >> 1;
    build(l, mid, num << 1);
    build(mid + 1, r, num << 1 | 1);
    push_up(num);
}
bool in(int l, int r, int L, int R)
{
    return (l >= L) && (r <= R);
}
bool out(int l, int r, int L, int R)
{
    return (l > R) || (r < L);
}
void make_tag(int num, int k, int kk, int kkk) //
{
    t[num].mx = max(t[num].mx, kk);
    t[num].mn = min(t[num].mn, kkk);
    t[num].date += k * (t[num].r - t[num].l + 1);
    t[num].lz += k;
    t[num].lx = max(t[num].lx, kk);
    t[num].ln = min(t[num].ln, kkk);
}
void push_down(int num)
{
    int mid = (t[num].l + t[num].r) >> 1;
    make_tag(num << 1, t[num].lz, t[num].lx, t[num].ln);
    make_tag(num << 1 | 1, t[num].lz, t[num].lx, t[num].ln);
    t[num].lz = 0;
    t[num].lx = -inf;
    t[num].ln = inf;
}
void up_date(int l, int r, int k, int num)
{
    if (out(l, r, t[num].l, t[num].r))
        return;
    if (in(t[num].l, t[num].r, l, r))
    {
        make_tag(num, k, k, k);
        return;
    }
    push_down(num);
    up_date(l, r, k, num << 1);
    up_date(l, r, k, (num << 1) + 1);
    push_up(num);
}
void ans_query(int num)
{
    ans.mx = max(ans.mx, t[num].mx);
    ans.mn = min(ans.mn, t[num].mn);
    ans.date += t[num].date;
}
void query(int l, int r, int num) // 查询区间的l,r不会变，只有子树的区间会变
{
    if (in(t[num].l, t[num].r, l, r))
    {
        ans_query(num);
        return;
    }
    if (out(t[num].l, t[num].r, l, r))
    {
        return;
    }
    int mid = (t[num].l + t[num].r) >> 1;
    push_down(num);
    if (l <= mid)
        query(l, r, num << 1);
    if (r > mid)
        query(l, r, num << 1 | 1);
    return;
}
void find(int l, int r, int num)
{
    int mid = (l + r) >> 1;
    if (t[num].l == 0 || t[num].r == 0)
        return;
    cout << t[num].l << " " << t[num].r << " " << t[num].date << " " << t[num].mx << " " << t[num].mn << "\n";
    if (l == r)
    {
        return;
    }
    find(num << 1, l, mid);
    find(num << 1 | 1, mid + 1, r);
    return;
}
signed main()
{
    cin >> n;
    int q;
    cin >> q;
    for (int i = 1; i <= n; i++)
        cin >> p[i];
    build(1, n, 1);
    // find(1, n, 1);
    return 0;
}
```
### 关键
{
在查询和结点更改的时候，查询区间l,r不会改变，in的信息是查询区间l,r包含树的区间
}

