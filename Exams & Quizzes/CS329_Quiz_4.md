## CS329 Machine Learning(H) Quiz 4

### Question 1

For a GMM, $D = \{x_1,\dots,x_N\}$, $\theta = \{\pi_k,\mu_k,\Sigma_k\}_{k=1}^K$

1. What is the ML solution of $\theta$ ?
2. If $\pi\sim\text{Dir}(N_{10},\dots,N_{K0})$, $\mu_k\sim\mathcal N(\mu_k\vert m_{k0},\Sigma_{k0})$. What is the MAP solution of $\theta$ ?
3. What is $p(x_{N+1}\vert\theta_\text{MAP})$ ?

#### Solution 1.1

The GMM: 
$$
p(x_i \vert \theta) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i \vert \mu_k, \Sigma_k)
$$
The log likelihood function:
$$
\ln p(D\vert \pi,\mu,\Sigma) = \sum\limits_{n=1}^N \ln \left\{\sum\limits_{k=1}^K\pi_k\mathcal N(x_n
\vert \mu_k,\Sigma_k)\right\}
$$

Derivative with respect to $\pi_k$:
$$
\frac{\partial}{\partial \pi_k} \ln p(D \vert \pi, \mu, \Sigma) = \sum_{n=1}^N \frac{\mathcal N(x_n \vert \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal N(x_n \vert \mu_j, \Sigma_j)}
$$
Setting the derivative equal to zero:

$$
\sum_{n=1}^N \frac{\mathcal N(x_n \vert \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal N(x_n \vert \mu_j, \Sigma_j)} = 0
$$
Derivative with respect to $\mu_k$:
$$
 \frac{\partial}{\partial \mu_k} \ln p(D \vert \pi, \mu, \Sigma) = \sum_{n=1}^N \frac{\pi_k \mathcal N(x_n \vert \mu_k, \Sigma_k) \Sigma_k^{-1} (x_n - \mu_k)}{\sum_{j=1}^K \pi_j \mathcal N(x_n \vert \mu_j, \Sigma_j)} 
$$
Setting the derivative equal to zero:

$$
\sum_{n=1}^N \frac{\pi_k \mathcal N(x_n \vert \mu_k, \Sigma_k) \Sigma_k^{-1} (x_n - \mu_k)}{\sum_{j=1}^K \pi_j \mathcal N(x_n \vert \mu_j, \Sigma_j)} = 0
$$
Derivative with respect to $\Sigma_k$:
$$
\frac{\partial}{\partial \Sigma_k} \ln p(D \vert \pi, \mu, \Sigma) = \sum_{n=1}^N \frac{\pi_k \mathcal N(x_n \vert \mu_k, \Sigma_k) \Sigma_k^{-1} [(x_n - \mu_k)(x_n - \mu_k)^\text T - \Sigma_k]}{2 \sum_{j=1}^K \pi_j \mathcal N(x_n \vert \mu_j, \Sigma_j)}
$$
Setting the derivative equal to zero:

$$
\sum_{n=1}^N \frac{\pi_k \mathcal N(x_n \vert \mu_k, \Sigma_k) \Sigma_k^{-1} [(x_n - \mu_k)(x_n - \mu_k)^\text T - \Sigma_k]}{\sum_{j=1}^K \pi_j \mathcal N(x_n \vert \mu_j, \Sigma_j)} = 0
$$
For the non-linear system above, we can then use EM methods to solve the ML solution of $\theta$:

1. Initialize $\theta$, and evaluate the initial value of the log likelihood.

2. **E step**. Evaluate the responsibilities using the current parameter values
   $$
   \gamma(z_{nk})=\frac{\pi_k\mathcal N(x_n \vert \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal N(x_n \vert \mu_j, \Sigma_j)}
   $$
   
3. **M step**. Re-estimate the parameters using the current responsibilities
   $$
   \begin{align*}
   \mu_k^\text{new} &= \frac 1 {N_k} \sum\limits_{n=1}^N \gamma(z_{nk}) x_n\\
   \Sigma_k^\text{new} &= \frac 1 {N_k} \sum\limits_{n=1}^N \gamma(z_{nk}) (x_n-\mu_k^\text{new})(x_n-\mu_k^\text{new})^\text T\\
   \pi_k^\text{new} &=\frac 1 N \sum\limits_{n=1}^N \gamma(z_{nk})=  \frac {N_k} N
   \end{align*}
   $$

4. Evaluate the log likelihood function
   $$
   \ln p(D\vert \pi,\mu,\Sigma) = \sum\limits_{n=1}^N \ln \left\{\sum\limits_{k=1}^K\pi_k\mathcal N(x_n
   \vert \mu_k,\Sigma_k)\right\}
   $$
   and check for convergence of either the parameters or the log likelihood. If the convergence criterion is not satisfied return to step 2.

#### Solution 1.2

To solve $\pi_k$,
$$
\sum_{n=1}^N \frac{\mathcal N(x_n | \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal N(x_n | \mu_j, \Sigma_j)} = \frac{N_k + N_{k0}}{\pi_k}
$$
To solve $\mu_k$,
$$
\sum_{n=1}^N \frac{\pi_k \mathcal N(x_n | \mu_k, \Sigma_k) \Sigma_k^{-1} (x_n - \mu_k)}{\sum_{j=1}^K \pi_j \mathcal N(x_n | \mu_j, \Sigma_j)} = \Sigma_{k0}^{-1} (\mu_k - m_{k0})
$$
To solve $\Sigma_k$,
$$
\sum_{n=1}^N \pi_k \gamma(z_{nk})[(x_n - \mu_k)(x_n - \mu_k)^\text T] = \Sigma_k (\Sigma_{k0}^{-1} + \Sigma_k^{-1})
$$
Similar to 1.1, but the **M step** is modified for MAP:

**M step**. Re-estimate the parameters using the current responsibilities
$$
\begin{align*}
\mu_k^\text{new} &= (\Sigma_k\Sigma_{k0}^{-1} + \frac 1 {N_k} \sum\limits_{n=1}^N \gamma(z_{nk}))^{-1}(\frac 1 {N_k} \sum\limits_{n=1}^N \gamma(z_{nk})x_n + \Sigma_k\Sigma_{k0}^{-1}m_{k0})\\
\Sigma_k^\text{new} &= (\frac 1 {N_k} \sum\limits_{n=1}^N (\gamma(z_{nk}) (x_n-\mu_k^\text{new})(x_n-\mu_k^\text{new})^\text T) - I)\Sigma_{k0}\\
\pi_k^\text{new} &=\frac 1 {N+N_{0}} \sum\limits_{n=1}^N \gamma(z_{nk})=  \frac {N_k + N_{k0}} {N+N_0}
\end{align*}
$$
Then by EM method we can obtain $\theta_\text{MAP}$.

#### Solution 1.3

The prediction:
$$
p(x_i \vert \theta^\text{MAP}) = \sum_{k=1}^{K} \pi_k^\text{MAP} \mathcal{N}(x_{N+1} \vert \mu_k^\text{MAP}, \Sigma_k^\text{MAP})
$$

### Question 2

For a HMM, $D = \{x_1,\dots,x_N\}$, $\theta = \{\pi,A,\mu_k,\Sigma_k\}_{k=1}^K$

1. What is the ML solution of $\theta$ ?
2. If $\pi\sim\text{Dir}(N_{10},\dots,N_{K0})$, $A^{(k)}\sim \text{Dir}(M_{10}^{(k)},M_{10}^{(k)})$, $\mu_k\sim\mathcal N(\mu_k\vert m_{k0},\Sigma_{k0})$. What is the MAP solution of $\theta$ ?
3. What is $p(x_{N+1}\vert\theta_\text{MAP})$ ?

#### Solution 2.1

The likelihood function
$$
p(D\vert \theta) = \sum\limits_Z(D,Z\vert\theta)
$$
EM algorithm to find an efficient framework
$$
Q(\theta,\theta^\text{old}) = \sum\limits_{k=1}^K \gamma (z_{1k})\ln \pi_k + \sum\limits_{n=2}^N\sum\limits_{j=1}^K\sum\limits_{k=1}^K\xi(z_{(n-1)j},z_{nk})\ln A_{jk} + \sum\limits_{n=1}^N\sum\limits_{j=1}^K \gamma(z_{nk}\ln p(x_n\vert \phi_k))
$$

1. Initialize $\theta$, and evaluate the initial value of the log likelihood.

2. **E step**. Use $\gamma(z_n)$ to denote the marginal posterior distribution of a latent variable $z_n$, and $\xi(z_{n−1}, z_n)$ to denote the joint posterior distribution of two successive latent variables, evaluate these quantities in this step:
   $$
   \begin{align*}
   \gamma(z_n) &= p(z_n\vert D,\theta^\text{old})\\
   \xi(z_{n-1},z_n) &= p(z_{n-1},z_n\vert D,\theta^\text{old})\\
   \gamma(z_{nk}) &= \mathbb E[z_{nk}] = \sum\gamma(z)z_{nk}\\
   \xi(z_{(n-1)j},z_{nk})&= \mathbb E[z_{(n-1)j}z_{nk}] = \sum\limits_{z}\gamma(z)z_{(n-1)j}z_{nk} \\
   \end{align*}
   $$

3. **M step**. Re-estimate the parameters using the current responsibilities
   $$
   \begin{align*}
   \pi_k^\text{new} &= \frac{\gamma(z_{1k})}{\sum\limits_{j=1}^K \gamma(z_{1j})}\\
   A_{jk}^\text{new} &= \frac{\sum\limits_{n=2}^N \xi(z_{(n-1)j},z_{nk}) }{\sum\limits_{l=1}^K\sum\limits_{n=2}^N \xi(z_{(n-1)j,z_{nl}})}\\
   \mu_k^\text{new} &= \frac{\sum\limits_{n=1}^N \gamma(z_{nk})x_n}{\sum\limits_{n=1}^N \gamma(z_{nk})}\\
   \Sigma_{k}^\text{new} &= \frac{\sum\limits_{n=1}^N\gamma(z_{nk})(x_n-\mu_k)(x_n-\mu_k)^\text T}{\sum\limits_{n=1}^N\gamma(z_{nk})}
   \end{align*}
   $$

4. Evaluate the log likelihood function and check for convergence of either the parameters or the log likelihood. If the convergence criterion is not satisfied return to step 2.

#### Solution 2.2

Modify the **M step** of 2.1 to obtain the MAP solution:
$$
\begin{align*} \pi_k^\text{new} &= \frac{N_{k0}+\gamma(z_{1k})}{N_0+\sum\limits_{j=1}^K\gamma(z_{1j})}\\ A_{jk}^\text{new} &= \frac{M_{0,jk}+\sum\limits_{n=2}^N \xi(z_{(n-1)j},z_{nk})}{\sum\limits_{l=1}^K M_{0,jl}+\sum\limits_{l=1}^K\sum\limits_{n=2}^N \xi(z_{(n-1)j,z_{nl}})}\\ \mu_k^\text{new} &= \frac{\sum\limits_{n=1}^N \gamma(z_{nk})x_n+\Sigma_{k0}^{-1}m_{k0}}{N+\Sigma_{k0}^{-1}}\\ \Sigma_k^\text{new} &= \frac{\sum\limits_{n=1}^N \gamma(z_{nk})(x_n-\mu_k)(x_n-\mu_k)^\text T+\Sigma_{k0}^{-1}}{N+\Sigma_{k0}^{-1}} \end{align*}
$$

Then by EM method we can obtain $\theta_\text{MAP}$.

#### Solution 2.3

The prediction:
$$
p(x_{N+1}\vert \theta_\text{MAP}) = \sum\limits_{z_{N+1}=1}^K p(x_{N+1},z_{N+1}\vert \theta_\text{MAP}) = \sum\limits_{z_{N+1}=1}^K \int p(x_{N+1}\vert \phi_{z_{N+1}})p(\phi_{z_{N+1}}\vert \theta_\text{MAP})d\phi_{z_{N+1}}
$$

### Question 3

For a stock market model, $\pi = [0.5,0.5]$, $A=\begin{bmatrix}0.6 & 0.3 \\ 0.4 &0.7\end{bmatrix}$, $B = \begin{bmatrix}0.8&0.1\\0.2&0.9\end{bmatrix}$, $z=\{\text{bull},\text{bear}\}$, $D=\{\text{rise},\text{fall}\}$

If we have an observation of $D = \{\text{fall},\text{fall},\text{rise}\}$:

1. What is $p(D\vert \pi,A,B)$?
2. What are $p(z_2\vert D,\pi, A,B)$ and $p(z_2,z_3\vert D,\pi,A,B)$, respectively?
3. What is the optimal $\{z_1,z_2,z_3\}$?
4. If there is a new observation of $x_4 = \text{rise}$, what is $p(x_4\vert D,\pi,A,B)$?

#### Solution 3.1

$$
\begin{align*}
\alpha(z_1) &= p(z_1,x_1) = p(x_1\vert z_1)p(z_1)\\
\alpha(z_2) &= p(z_2,x_1,x_2)=p(x_2\vert z_2)\sum\limits_{z_1}p(z_2\vert z_1)\alpha(z_1)\\
\alpha(z_3) &= p(z_3,x_1,x_2,x_3)=p(x_3\vert z_3)\sum\limits_{z_2}p(z_3\vert z_2)\alpha(z_2)\\
\end{align*}
$$

Therefore
$$
\begin{align*}
p(x_1&=\text{fall},z_1 = \text{bull/bear}) = \begin{bmatrix}0.2\\0.9\end{bmatrix}\cdot \begin{bmatrix}0.5\\0.5\end{bmatrix} = \begin{bmatrix}0.1\\0.45\end{bmatrix}\\
p(x_2&=\text{fall},z_2 = \text{bull/bear}) = \begin{bmatrix}0.2\\0.9\end{bmatrix}\cdot \begin{bmatrix}0.6*0.1 + 0.3*0.45\\0.4*0.1 +0.7*0.45\end{bmatrix} = \begin{bmatrix}0.039\\0.3195\end{bmatrix}\\
p(x_3&=\text{rise},z_3 = \text{bull/bear}) = \begin{bmatrix}0.8\\0.1\end{bmatrix}\cdot \begin{bmatrix}0.6*0.039 + 0.3*0.3195\\0.4*0.039 +0.7*0.3195\end{bmatrix} = \begin{bmatrix}0.0954\\0.023925\end{bmatrix}\\
\end{align*}
$$
So we have
$$
p(D\vert \pi,A,B) = p(x_1,x_2,x_3) = \sum\limits_{z_3}p(z_3,x_1,x_2,x_3) = 0.0954 + 0.023925 = 0.119325
$$

#### Solution 3.2

$$
\begin{align*}
p(z_2\vert D,\pi, A,B) &= \frac{p(z_1,x_1,x_2)}{p(x_1,x_2)} = \begin{bmatrix}0.1088\\0.8912\end{bmatrix}\\
p(z_2,z_3\vert D,\pi, A,B) &= p(z_3\vert z_2)p(z_2\vert D,\pi, A,B) \\&=\begin{bmatrix}🐂🐂&🐂🐻\\🐻🐂&🐻🐻\end{bmatrix} \\&=\begin{bmatrix}0.8*0.1088&0.2*0.1088\\0.7*0.8912&0.3*0.8912\end{bmatrix}\\&=\begin{bmatrix}0.08704&0.02176\\0.62384&0.26736\end{bmatrix}
\end{align*}
$$

#### Solution 3.3

The optimal $\{z_1,z_2,z_3\}$ is $\{\text{bear},\text{bear},\text{bull}\}$, which maximizes $p(z_1,z_2,z_3\vert D,\pi,A,B)$.

#### Solution 3.4

$$
\begin{align*}
p(z_4\vert D,\pi,A,B) &= p(z_4\vert z_3)p(z_3\vert D,\pi,A,B) = \begin{bmatrix}0.0954*0.8+0.023925*0.3\\0.0954*0.2+0.023925*0.7\end{bmatrix}=\begin{bmatrix}0.0834975\\0.0358275\end{bmatrix}\\
p(x_4=\text{rise}\vert \pi,A,B) &= \sum\limits_{z_4}p(x_4\vert z_4,\pi,A,B)p(z_4\vert D,\pi,A,B) = 0.9*0.08349 + 0.4*0.03582 = 0.08947875
\end{align*}
$$
