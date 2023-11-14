## CS329 Machine Learning(H) Quiz 2

### Before Quiz

$$
\begin{align*}
&-(a_0-1)\ln\pi - (b_0-1)\ln(1-\pi) \quad &\text{prior of }\pi\\
&+\frac 1 2(\mu_1-m_{10})^\text T \Sigma_{10}^{-1}(\mu_1-m_{10}) \quad &\text{prior of }\mu_1\\
&+\frac 1 2(\mu_2-m_{20})^\text T \Sigma_{20}^{-1}(\mu_2-m_{20})\quad &\text{prior of }\mu_2\\
&+\sum\limits_{n=1}^N[t_n\ln\left(\pi\mathcal N (x\vert\mu_1,\Sigma_1)\right)+(1-t_n)\ln((1-\pi)\mathcal N (x\vert\mu_2,\Sigma_2))]\quad&\text{Likelihood}
\end{align*}
$$

---

### Question 1 Generative Gaussian Mixture

#### Question 1.1

$p(x)=\pi\mathcal N(x\vert\mu_1,\Sigma_1)+(1-\pi)\mathcal N(x\vert\mu_2,\Sigma_2)$

Given $\mathbf x=[x_1,\dots,x_N]$, $\mathbf t=[t_1,\dots,t_N]$

What's the ML estimation of $\mu_1,\Sigma_1,\mu_2,\Sigma_2,\pi$?

#### Solution 1.1

Derive log-likelihood function w.r.t. $\pi$,
$$
\frac{\partial}{\partial \pi}L=\frac{\partial}{\partial \pi}\sum\limits_{n=1}^N \{t_n\ln\pi+(1-t_n)\ln(1-\pi)\}=\frac{N_1}{\pi}+\frac{N_2}{\pi-1}=0
$$
Derive log-likelihood function w.r.t. $\mu_i$,
$$
\begin{align*}
\frac{\partial}{\partial \mu_1}L &= \frac{\partial}{\partial \mu_1} \sum\limits_{n=1}^N t_n\ln\mathcal N(x_n\vert,\mu_1,\Sigma_1)\\
&=\frac{\partial}{\partial \mu_1} -\frac 1 2 \sum\limits_{n=1}^N t_n(x_n-\mu_1)^\text T \Sigma_1^{-1}(x_n-\mu_1)\\
&=0\\
\frac{\partial}{\partial \mu_2}L &= \frac{\partial}{\partial \mu_2} \sum\limits_{n=1}^N (1-t_n)\ln\mathcal N(x_n\vert,\mu_2,\Sigma_2)\\
&=\frac{\partial}{\partial \mu_2} -\frac 1 2 \sum\limits_{n=1}^N (1-t_n)(x_n-\mu_2)^\text T \Sigma_1^{-1}(x_n-\mu_2)\\
&=0
\end{align*}
$$
Derive log-likelihood function w.r.t. $\Sigma_i$,
$$
\begin{align*}
\frac{\partial}{\partial \Sigma_1} L &= \frac{\partial}{\partial \Sigma_1} \left(-\frac 1 2 \sum\limits_{n=1}^N t_n\ln |\Sigma_1|-\frac1 2 \sum\limits_{n=1}^N t_n(x_n-\mu_1)^\text T \Sigma_1^{-1}(x_n-\mu_1)\right)=0\\
\frac{\partial}{\partial \Sigma_2} L &= \frac{\partial}{\partial \Sigma_2} \left(-\frac 1 2 \sum\limits_{n=1}^N (1-t_n)\ln |\Sigma_2|-\frac1 2 \sum\limits_{n=1}^N (1-t_n)(x_n-\mu_2)^\text T \Sigma_2^{-1}(x_n-\mu_2)\right)=0
\end{align*}
$$
Hence we have
$$
\begin{align*}
\pi_\text{ML} &= \frac 1 N \sum\limits_{n=1}^N t_n =\frac{N_1}{N}=\frac{N_1}{N_1+N_2}\\
\mu_{1\text{ML}} &= \frac 1 {N_1}\sum\limits_{n=1}^Nt_n x_n\\
\mu_{2\text{ML}} &= \frac 1 {N_2}\sum\limits_{n=1}^N(1-t_n) x_n\\
\Sigma_{1\text{ML}} &=\frac 1 {N_1}\sum\limits_{x_n\in\mathcal C_1} (x_n-\mu_{1\text{ML}})(x_n-\mu_{1\text{ML}})^\text T\\
\Sigma_{2\text{ML}} &=\frac 1 {N_2}\sum\limits_{x_n\in\mathcal C_2} (x_n-\mu_{2\text{ML}})(x_n-\mu_{2\text{ML}})^\text T
\end{align*}
$$

where

- $N_1$ is the number of data points in class 1,
- $N_2$ is the number of data points in class 2.

#### Question 1.2

$\pi\sim beta(a_0,b_0), p(\mu_i) = \mathcal N(m_{i0},\Sigma_{i0}), i=1,2$

Given $\mathbf x=[x_1,\dots,x_N]$, $\mathbf t=[t_1,\dots,t_N]$

What's the MAP estimation of $\mu_1,\Sigma_1,\mu_2,\Sigma_2,\pi$?

#### Solution 1.2

Posterior of $\pi$:
$$
\begin{align*}
p(\pi|\mathbf{t}) &\propto p(\mathbf{t}|\pi) \cdot p(\pi)\\
&= \pi^{N_1} \cdot (1 - \pi)^{N_2} \cdot \pi^{a_0-1} \cdot (1-\pi)^{b_0-1}\\
&= \pi^{N_1 + a_0 - 1} \cdot (1 - \pi)^{N_2 + b_0 - 1}
\end{align*}
$$
By using the property of product of Gaussian distributions, we obtain the MAP estimation of $\mu_i$ and $\Sigma_i$
$$
\begin{align*}
\pi_\text{MAP} &= \frac{N_1+a_{0}-1}{N+a_0+b_0-2}=\frac{N_1+a_0-1}{N_1+N_2+a_0+b_0-2}\\
\Sigma_{1\text{MAP}} &=({\Sigma_{1\text{ML}}}^{-1}+{\Sigma_{10}}^{-1})^{-1}\\
\Sigma_{2\text{MAP}} &=({\Sigma_{2\text{ML}}}^{-1}+{\Sigma_{20}}^{-1})^{-1}\\
\mu_{1\text{MAP}} &= \Sigma_{1\text{MAP}}({\Sigma_{1\text{ML}}}^{-1}\mu_{1\text{ML}}+{\Sigma_{10}}^{-1}m_{10})\\
\mu_{2\text{MAP}} &= \Sigma_{2\text{MAP}}({\Sigma_{2\text{ML}}}^{-1}\mu_{2\text{ML}}+{\Sigma_{20}}^{-1}m_{20})\\
\end{align*}
$$

#### Question 1.3

What's $p(\mathcal C_1\vert x)$ for ML and MAP models respectively?

#### Solution 1.3

By Bayes' Theorem,
$$
\begin{align*}
p_\text{ML}(\mathcal C_1\vert x)&=\frac{p_\text{ML}(x,\mathcal C_1)}{p_\text{ML}(x)}=\frac{\pi_\text{ML}\mathcal N(x\vert\mu_\text{1ML},\Sigma_\text{1ML})}{\pi_\text{ML}\mathcal N(x\vert\mu_\text{1ML},\Sigma_\text{1ML})+(1-\pi_\text{ML})\mathcal N(x\vert\mu_\text{2ML},\Sigma_\text{2ML})}\\
p_\text{MAP}(\mathcal C_1\vert x)&=\frac{p_\text{MAP}(x,\mathcal C_1)}{p_\text{MAP}(x)}=\frac{\pi_\text{MAP}\mathcal N(x\vert\mu_\text{1MAP},\Sigma_\text{1MAP})}{\pi_\text{MAP}\mathcal N(x\vert\mu_\text{1MAP},\Sigma_\text{1MAP})+(1-\pi_\text{MAP})\mathcal N(x\vert\mu_\text{2MAP},\Sigma_\text{2MAP})}
\end{align*}
$$

---

### Question 2 Discriminative Logistic Regression

#### Question 2.1

$y=\sigma(w^\text T\phi(x))$

Given $\mathbf x=[x_1,\dots,x_N]$, $\mathbf t=[t_1,\dots,t_N]$

What's the ML estimation of $q(w)$?

#### Solution 2.1

By Gauss-Newton iteration: $w^\text{new}=w^\text{old}-H^{-1}\nabla E(w)$, we obtain $w_\text{ML}$,

where
$$
\begin{align*}
E(w) &= -\sum\limits_{n=1}^N  [t_n\ln y_n + (1-t_n)\ln(1-y_n)]\\
\nabla E(w) &= \sum\limits_{n=1}^N (y_n-t_n)\phi(x_n)\\
\text{step size}\quad H &= \nabla^2 E(w) = \sum\limits_{n=1}^N y_n(1-y_n)\phi(x_n)\phi(x_n)^\text T
\end{align*}
$$
Hence
$$
q(w) = \mathcal N(w\vert w_\text{ML}, H^{-1})
$$

#### Question 2.2

$y=\sigma(w^\text T\phi(x))$, $p(w)\sim\mathcal N(m_0,\Sigma_0)$

Given $\mathbf x=[x_1,\dots,x_N]$, $\mathbf t=[t_1,\dots,t_N]$

What's the MAP estimation of $q(w)$?

#### Solution 2.2

By Gauss-Newton iteration: $w^\text{new}=w^\text{old}-H^{-1}\nabla E(w)$, we obtain $w_\text{ML}$,

where
$$
\begin{align*}
E(w) &= \frac{1}{2} (w-m_0)^\text T\Sigma_0^{-1}(w-m_0)-\sum\limits_{n=1}^N  [t_n\ln y_n + (1-t_n)\ln(1-y_n)]\\
\nabla E(w) &=\Sigma_0^{-1}(w-m_0)+ \sum\limits_{n=1}^N (y_n-t_n)\phi(x_n)\\
\text{step size}\quad H &= \nabla^2 E(w) = \Sigma_0^{-1}+\sum\limits_{n=1}^N y_n(1-y_n)\phi(x_n)\phi(x_n)^\text T
\end{align*}
$$
Hence 
$$
q(w) = \mathcal N(w\vert w_\text{MAP}, H^{-1})
$$


#### Question 2.3

What's $p(t\vert y(w,x))$ for ML and MAP estimation, respectively?

#### Solution 2.3

Probability of $t=1$ w.r.t. ML and MAP estimations
$$
\begin{align*}
p_\text{ML}(t=1\vert y(w_\text{ML},x)) &= \sigma(w_\text{ML}^\text T\phi(x)) = \frac{1}{1+\exp(-w_\text{ML}^\text T\phi(x))}\\
p_\text{MAP}(t=1\vert y(w_\text{MAP},x)) &= \sigma(w_\text{MAP}^\text T\phi(x))= \frac{1}{1+\exp(-w_\text{MAP}^\text T\phi(x))}\\
\end{align*}
$$
