## CS329 Machine Learning(H) Quiz 1

### Question 1

Given $x=u+v$, $u\sim\mathcal N(m_0,\Sigma_0)$,$v\sim\mathcal N(0,Q)$, $D=[x_1,\cdots,x_N]$

1. $p(u\vert D,m_0,\Sigma_0,Q)$
2. $p(x\vert D,m_0,\Sigma_0,Q)$
3. evidence $p(D\vert m_0,\Sigma_0,Q)$

### Solution 1.1

prior: $x\sim\mathcal N(m_0,\Sigma_0+Q)$

$p(u | D, m_0, \Sigma_0, Q) \propto p(D | m_0, \Sigma_0, Q) \cdot p(u | m_0, \Sigma_0).$

$p(D | m_0, \Sigma_0, Q)=\frac{1}{(2\pi (\Sigma_0+Q))^{N/2}}\exp\left\{-\frac{1}{2(\Sigma_0+Q)}\sum\limits_{n=1}^N (x_n-m_0)^2\right\}$

$p(u\vert m_0,\Sigma_0)=\frac{1}{(2\pi \Sigma_0)^{1/2}}\exp\left\{-\frac{1}{2\Sigma_0} (u-m_0)^2\right\}$

Hence Posterior Distribution $p(u | D, m_0, \Sigma_0, Q) \sim \mathcal N(m_u, \Sigma_u),$

where:

- $\Sigma_u^{-1} = NQ^{-1}+\Sigma_0^{-1}$
- $\Sigma_u^{-1}m_u=Q^{-1}\sum\limits_{n=1}^Nx_n+\Sigma_0^{-1}m_0$

### Solution 1.2

$x=u+v$

$p(x\vert D,m_0,\Sigma_0,Q)=\mathcal N(m_u,\Sigma_u+Q)$

### Solution 1.3

$$
\begin{align*}
p(D\vert m_0,\Sigma_0,Q)&=\int p(D\vert u,Q)p(u\vert m_0,\Sigma_0) du\\
&=\frac{1}{(2\pi)^{N/2}\vert\Sigma_0\vert^{\frac 1 2}}\int \exp\left\{-\frac 1 2 \sum\limits_{n=1}^N(x_n-u)^\text TQ^{-1}(x_n-u)-\frac 1 2 \sum\limits_{n=1}^N(u-m_0)^\text T\Sigma_0^{-1}(u-m_0)\right\} du\\
&=\frac{1}{(2\pi)^{N/2}\vert\Sigma_0\vert^{\frac 1 2}}\int \exp\left\{\right\}
\end{align*}
$$

### Question 2

Given $y=Ax+v$, $x\sim\mathcal N(m_0,\Sigma_0)$,$v\sim\mathcal N(0,Q)$, $D=[y_1,\cdots,y_N]$

1. $p(x\vert D,m_0,\Sigma_0,Q)$
2. $p(y\vert D,m_0,\Sigma_0,Q)$
3. evidence $p(D\vert m_0,\Sigma_0,Q)$

### Solution 2.1

$p(x | D, m_0, \Sigma_0, Q) \propto p(D | m_0, \Sigma_0, Q) \cdot p(x | m_0, \Sigma_0).$

$p(D | m_0, \Sigma_0, Q) = \prod\limits_{n=1}^N p(x_n)=p(D | m_0, \Sigma_0, Q)= \frac{1}{(2\pi\Sigma_0)^{N/2}}\prod\limits_{n=1}^N   \exp\left\{-\frac{1}{2\Sigma_0}(x_n - m_0)^2\right\}.$

$p(x | m_0, \Sigma_0)=\mathcal N(m_0,\Sigma_0)$

Hence Posterior Distribution $p(x | D, m_0, \Sigma_0, Q) \sim \mathcal N(m_x, \Sigma_x),$

where:

- $\Sigma_x = \left(\Sigma_0^{-1} + NA^TQ^{-1}A\right)^{-1}$

- $\Sigma_x^{-1}m_x=\sum\limits_{n=1}^NA^TQ^{-1}y_n+\Sigma_0^{-1}m_0$

### Solution 2.2

$y=Ax+v$.

$p(y | D, m_0, \Sigma_0, Q) =\mathcal N(Am_x, A\Sigma_xA^T + Q)$

### Solution 2.3



### Review

#### Learning

$$
p(\theta\vert\mathcal D)\propto p(\mathcal D\vert\theta)\cdot p(\theta)
$$

#### Prediction

$$
p(t\vert x,\mathcal D)=\int p(t\vert x,\mathcal \theta) \cdot p(\theta\vert\mathcal D) \text d \theta
$$

#### Evidence

$$
p(\mathcal D)=\int p(\mathcal D\vert\theta)\cdot p(\theta) \text d\theta
$$

