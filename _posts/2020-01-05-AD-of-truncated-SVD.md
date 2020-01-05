---
layout: page
title: Automatic differentiation of truncated SVD
typora-root-url: /home/hendry/Documents/buwantaiji.github.io/
---
<script type="text/javascript" async src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>

<script type="text/x-mathjax-config">   MathJax.Hub.Config({     TeX: { equationNumbers: { autoNumber: "all" } }   }); </script>
Truncated SVD (Singular Value Decomposition) is an important linear algebra operation that has wide applications in various tensor network algorithms. In this article, we will study the (reverse mode) automatic differentiation of truncated SVD. Due to the close relation between SVD and eigen-decomposition, it turns out that one is faced with similar difficulties when trying to derive the back-propagation formulas for truncated SVD and dominant eigensolver. In view of this, the presentation here is pretty parallel to that of dominant eigensolver in previous articles. In particular, the adjoint method and the approach based on full SVD are used to derive the desired results. These two approaches are completely equivalent and reveal the mechanism behind the scene that effectively separates the information we want out of the full spectrum.

$\renewcommand{\vec}[1]{\boldsymbol{#1}}$

## Some mathematical backgrounds of SVD

---

The study of automatic differentiation of truncated SVD through the adjoint method requires some detailed mathematical understanding of SVD. In this section, we will give necessary backgrounds in this regard.

Let $A$ be an arbitrary $M \times N$ real matrix, where $M$ is generally not equal to $N$. We will derive the SVD of $A$ in a constructive way. It turns out that the SVD of $A$ is intimately related to the eigen-decomposition of the square matrices $A A^T$ and $A^T A$, both of which are symmetric and positive semi-definite.  This implies that the eigenvalues of both $A A^T$ and $A^T A$ are non-negative. Furthermore, their eigen-spectrums have close relations, which can be seen from the following propositions:

>**Proposition 1**: Let $A$ be an arbitrary $M \times N$ real matrix.
>
>1. If an $M$-dimensional vector $\vec{l}$ is the eigenvector of $A A^T$ with eigenvalue $\lambda > 0$, then $A^T \vec{l} \neq \vec{0}$ is the eigenvector of $A^T A$ with the same eigenvalue $\lambda$; if $\vec{l}$ is the eigenvector of $A A^T$ with eigenvalue zero, then $A^T \vec{l} = \vec{0}$.
>2. If an $N$-dimensional vector $\vec{r}$ is the eigenvector of $A^T A$ with eigenvalue $\lambda > 0$, then $A \vec{r} \neq \vec{0}$ is the eigenvector of $A A^T$ with the same eigenvalue $\lambda$; if $\vec{r}$ is the eigenvector of $A^T A$ with eigenvalue zero, then $A \vec{r} = \vec{0}$.



> **Proposition 2**: Let $A$ be an arbitrary $M \times N$ real matrix.
>
> 1. $A A^T$ and $A^T A$ have the same set of non-zero (positive) eigenvalues, including their multiplicities. Let these eigenvalues be denoted as $(\lambda_1, \cdots, \lambda_k)$, one can write
>
>    
>    $$
>    A A^T \vec{l}_i = \lambda_i \vec{l}_i, \forall i = 1, \cdots, k; \quad A A^T \vec{l}_i = \vec{0}, \forall i = k+1, \cdots, M.
>    \label{eq: leigenvector}
>    $$
>
>    $$
>    A^T A \vec{r}_i = \lambda_i \vec{r}_i, \forall i = 1, \cdots, k; \quad A^T A \vec{r}_i = \vec{0}, \forall i = k+1, \cdots, N.
>    \label{eq: reigenvector}
>    $$
>
>    where $k \leq \min(M, N)$ is the total number of nonzero eigenvalues (including multiplicities), $\lambda_1 \geq \cdots \geq \lambda_k > 0$. $(\vec{l}_1, \cdots, \vec{l}_M)$ and $(\vec{r}_1, \cdots, \vec{r}_N)$ are the complete set of eigenvectors of $A A^T$ and $A^T A$, respectively.
>
> 2. Furthermore, the various eigenvectors in 1 can be chosen to satisfy the following conditions:
>
>    
>    $$
>    \vec{l}_i^T \vec{l}_j = \delta_{ij}, \forall i, j = 1, \cdots, M. \quad
>    \vec{r}_i^T \vec{r}_j = \delta_{ij}, \forall i, j = 1, \cdots, N.
>    \label{eq: normalization}
>    $$
>
>    $$
>    A^T \vec{l}_i = \sqrt{\lambda_i} \vec{r}_i, \forall i = 1, \cdots, k; A^T \vec{l}_i = \vec{0}, \forall i = k+1, \cdots, M.
>    \label{eq: from l to r}
>    $$
>
>    $$
>    A \vec{r}_i = \sqrt{\lambda_i} \vec{l}_i, \forall i = 1, \cdots, k; A \vec{r}_i = \vec{0}, \forall i = k+1, \cdots, N.
>    \label{eq: from r to l}
>    $$

The proofs are pretty easy and can be left as an exercise. **Proposition 2** is the direct consequence of **Proposition 1**. Note that generally the square matrices $A A^T$ and $A^T A$ can have different dimensions. Nevertheless, they have exactly the same set of nonzero eigenvalues, which is amazing. In discussions below, we will continue to employ the notations in **Proposition 2**, and the relevant properties Eqs. $\eqref{eq: leigenvector}$-$\eqref{eq: from r to l}$ will also be extensively used.

From Eqs. $\eqref{eq: normalization}$-$\eqref{eq: from r to l}$, it is easy to derive that for $i = 1, \cdots, M$, $j = 1, \cdots, N$, we have


$$
\vec{l}_i^T A \vec{r}_j = 
\begin{cases}
	\sqrt{\lambda_i} \delta_{ij}. & i, j = 1, \cdots, k. \\
	0. & \textrm{otherwise.}
\end{cases}
\label{eq: lAr}
$$


By introducing the following notations


$$
\tilde{U} = \begin{pmatrix}
				\vert & & \vert \\
				\vec{l}_1 & \cdots & \vec{l}_M \\
				\vert & & \vert
			\end{pmatrix}, 
\tilde{V} = \begin{pmatrix}
				\vert & & \vert \\
				\vec{r}_1 & \cdots & \vec{r}_N \\
				\vert & & \vert
			\end{pmatrix}, 
S = \begin{pmatrix}
		\sqrt{\lambda_1} \\
		& \ddots \\
		& & \sqrt{\lambda_k}.
	\end{pmatrix},
$$


one can write Eq. $\eqref{eq: lAr}$ in a more compact form as follows:


$$
\tilde{U}^T A \tilde{V} = 
\begin{pmatrix}
S \\
&
\end{pmatrix}_{M \times N}.
\label{eq: UAV}
$$


As indicated by the orthonormality condition $\eqref{eq: normalization}$, both $\tilde{U}$ and $\tilde{V}$ are orthogonal matrices. One can thus obtain from $\eqref{eq: UAV}$ that


$$
A = \tilde{U}
\begin{pmatrix}
S \\
&
\end{pmatrix}_{M \times N} \tilde{V}^T
= U S V^T,
\label{eq: SVD}
$$


where we have introduced the matrices


$$
U_{M \times k} = \begin{pmatrix}
				\vert & & \vert \\
				\vec{l}_1 & \cdots & \vec{l}_k \\
				\vert & & \vert
			\end{pmatrix}, 
V_{N \times k} = \begin{pmatrix}
				\vert & & \vert \\
				\vec{r}_1 & \cdots & \vec{r}_k \\
				\vert & & \vert
			\end{pmatrix}.
$$


The columns of $U$ and $V$ consist of the eigenvectors with nonzero eigenvalues of $A A^T$ and $A^T A$, respectively. Eq. $\eqref{eq: SVD}$ is the celebrated SVD (Singular Value Decomposition) of an arbitrary rectangle matrix $A$. The diagonal elements of  $S$, $\sqrt{\lambda_i} \equiv s_i > 0, \forall i = 1, \cdots k$, are usually called the (nonzero) **singular values** of the matrix $A$, while $(\vec{l}_1, \cdots, \vec{l}_k)$ and $(\vec{r}_1, \cdots, \vec{r}_k)$ are the corresponding **left singular vectors** and **right singular vectors**, respectively. 

## Back-propagation of truncated SVD: adjoint method

---

In this section, we will apply the adjoint method to the automatic differentiation of truncated SVD. The basic setting of the adjoint method can be found in previous articles. Specifically, Let $A = A(\vec{\theta})$ be a general $M \times N$ real matrix that depends on some input parameters $\vec{\theta} = (\theta_1, \cdots, \theta_P)$. For simplicity, consider only one certain singular value $s$ and corresponding left and right singular vector $\vec{l}$, $\vec{r}$, respectively. The case of multiple singular values and singular vectors can then be easily obtained. The output $(\vec{l}, \vec{r}, s)$ is effectively ($M+N+1$)-dimensional, and with the background knowledge developed in the last section, we choose the constraint functions as follows:


$$
f_i(\vec{l}, \vec{r}, s, \vec{\theta}) = 
\begin{cases}
	\left( A A^T - s^2 I_M \right)_i^T \vec{l}. & i = 1, \cdots, M. \\
	\left( A^T A - s^2 I_N \right)_i^T \vec{r}. & i = M+1, \cdots, M+N. \\
	\vec{l}^T A \vec{r} - s. & i = 0.
\end{cases}
\label{eq: fs for truncated SVD}
$$


where the notation $O_i$ means the $i$th column of the matrix $O$. The condition $f_0(\vec{l}, \vec{r}, s, \vec{\theta}) = 0$ imposes a certain kind of "normalization" constraint. To see its origin, simply note Eq. $\eqref{eq: SVD}$ implies that $U^T A V = S$, since $U^T U = V^T V = I_k$. By focusing on the diagonal element of this equation corresponding to the desired singular value, one immediately obtain the desired condition, that is, $\vec{l}^T A \vec{r} = s$.

Making use of Eq. $\eqref{eq: fs for truncated SVD}$, one can readily derive the back-propagation formula of truncated SVD through the adjoint method. In current case, the general formulation of the adjoint method presented in previous articles reduces to the following form:


$$
\overline{\vec{x}} \rightarrow \begin{pmatrix}
    						    \overline{\vec{l}} \\
    						    \overline{\vec{r}} \\
    						    \overline{s}
    					    \end{pmatrix}, \quad
        \vec{\eta} \rightarrow \begin{pmatrix}
    									    \vec{\eta}_{\vec{l}} \\ \vec{\eta}_{\vec{r}} \\ \eta_s
    							\end{pmatrix}, \quad
        \frac{\partial f}{\partial \theta_\mu} \rightarrow
        \begin{pmatrix}
    	    \frac{\partial f_1}{\partial \theta_\mu} \\ \vdots \\
    	    \frac{\partial f_M}{\partial \theta_\mu} \\
    	    \frac{\partial f_{M+1}}{\partial \theta_\mu} \\ \vdots \\
    	    \frac{\partial f_{M+N}}{\partial \theta_\mu} \\
    	    \frac{\partial f_0}{\partial \theta_\mu}
        \end{pmatrix} = 
        \begin{pmatrix}
    	    \frac{\partial A}{\partial \theta_\mu} A^T \vec{l} + 
    	    A \frac{\partial A^T}{\partial \theta_\mu} \vec{l} \\
    	    \frac{\partial A^T}{\partial \theta_\mu} A \vec{r} + 
    	    A^T \frac{\partial A}{\partial \theta_\mu} \vec{r} \\
    	    \vec{l}^T \frac{\partial A}{\partial \theta_\mu} \vec{r}
        \end{pmatrix}.
$$

$$
\left( \frac{\partial f}{\partial \vec{x}} 	\right)^T \rightarrow
        \left(\begin{array}{ccc|ccc|c}
        	\vert & & \vert & \vert & & \vert & \vert \\
        	\left( \frac{\partial f_1}{\partial \vec{l}} \right)^T & \cdots & 
        	\left( \frac{\partial f_M}{\partial \vec{l}} \right)^T & 
        	\left( \frac{\partial f_{M+1}}{\partial \vec{l}} \right)^T & \cdots & 
        	\left( \frac{\partial f_{M+N}}{\partial \vec{l}} \right)^T & 
        	\left( \frac{\partial f_0}{\partial \vec{l}} \right)^T \\
        	\vert & & \vert & \vert & & \vert & \vert \\ \hline
        	\vert & & \vert & \vert & & \vert & \vert \\
        	\left( \frac{\partial f_1}{\partial \vec{r}} \right)^T & \cdots & 
        	\left( \frac{\partial f_M}{\partial \vec{r}} \right)^T & 
        	\left( \frac{\partial f_{M+1}}{\partial \vec{r}} \right)^T & \cdots & 
        	\left( \frac{\partial f_{M+N}}{\partial \vec{r}} \right)^T & 
        	\left( \frac{\partial f_0}{\partial \vec{r}} \right)^T \\
        	\vert & & \vert & \vert & & \vert & \vert \\ \hline
        	\frac{\partial f_1}{\partial s} & \cdots & 
        	\frac{\partial f_M}{\partial s} & 
        	\frac{\partial f_{M+1}}{\partial s} & \cdots & 
        	\frac{\partial f_{M+N}}{\partial s} & 
        	\frac{\partial f_0}{\partial s}
            \end{array}\right) = 
        \begin{pmatrix}
        	A A^T - s^2 I_M & \vec{0} & A \vec{r} \\
        	\vec{0} & A^T A - s^2 I_N & A^T \vec{l} \\
        	-2s\vec{l}^T & -2s\vec{r}^T & -1
        \end{pmatrix}.
$$



The so-called **adjoint equation** $\left( \frac{\partial f}{\partial \vec{x}} 	\right)^T \vec{\eta} = \overline{\vec{x}}$ thus reads:


$$
\left(A A^T - s^2 I_M\right) \vec{\eta}_\vec{l} + \eta_s A \vec{r} = \overline{\vec{l}}. \\
\left(A^T A - s^2 I_N\right) \vec{\eta}_\vec{r} + \eta_s A^T \vec{l} = \overline{\vec{r}}. \\
-2s \vec{l}^T \vec{\eta}_\vec{l} - 2s \vec{r}^T \vec{\eta}_\vec{r} - \eta_s = \overline{s}.
\label{eq: adjoint equation}
$$


Similar to the case of dominant eigensolver, one could first solve for $\eta_s$ from the first two equations of $\eqref{eq: adjoint equation}$ and obtains


$$
\eta_s = \frac{1}{s} \vec{l}^T \overline{\vec{l}}
	   = \frac{1}{s} \vec{r}^T \overline{\vec{r}}.
\label{eq: eta s}
$$


The equality of the last two quantities is not so evident. In fact, this is related to the remaining gauge freedom of the settings. More specifically, conditions $\eqref{eq: fs for truncated SVD}$ doesn't uniquely determine the singular vectors $\vec{l}$  and $\vec{r}$; in fact, it is invariant under the gauge transformation $\vec{l} \rightarrow c \vec{l}, \vec{r} \rightarrow \frac{\vec{r}}{c}$, where $c$ is an arbitrary nonzero scaling factor. However, simple inspection would indicate that the gradient computation result through back-propagation is actually gauge-independent. Thus, one can set freely the value of the scaling factor $c$ as he wants. The most convenient choice, of course, is such that the normalization $\vec{l}^T \vec{l} = 1$ holds. Then the following conditions (for the chosen gauge) can be immediately obtained: 


$$
\vec{l}^T \vec{l} = \vec{r}^T \vec{r} = 1, \quad 
A^T \vec{l} = s \vec{r}, \quad A \vec{r} = s \vec{l}.
\label{eq: additional gauge condition}
$$


Owing to the gauge independence, Eq.  $\eqref{eq: additional gauge condition}$ can be used for the remaining adjoint method derivations without loss of generality. It is also consistent with Eqs. $\eqref{eq: normalization}$-$\eqref{eq: from r to l}$ for the original purpose of deriving SVD. 

By making use of $\eqref{eq: eta s}$ and $\eqref{eq: additional gauge condition}$, the first two equations of $\eqref{eq: adjoint equation}$ can be rewritten as



$$
\left(A A^T - s^2 I_M\right) \vec{\eta}_\vec{l} = \left( I_M - \vec{l} \vec{l}^T \right) \overline{\vec{l}}. \\
\left(A^T A - s^2 I_N\right) \vec{\eta}_\vec{r} = \left( I_N - \vec{r} \vec{r}^T \right) \overline{\vec{r}}.
$$



The general solutions for the vectors $\vec{\eta}\_\vec{l}$ and $\vec{\eta}\_\vec{r}$ then have the following form, respectively:



$$
\vec{\eta}_\vec{l} = c_\vec{l} \vec{l} + \vec{\xi}_\vec{l}, \quad \textrm{where $\vec{\xi}_\vec{l}$ satisfies} \\
\left(A A^T - s^2 I_M\right) \vec{\xi}_\vec{l} = \left( I_M - \vec{l} \vec{l}^T \right) \overline{\vec{l}}, \quad \vec{l}^T \vec{\xi}_\vec{l} = 0.
\label{eq: xi l}
$$

$$
\vec{\eta}_\vec{r} = c_\vec{r} \vec{r} + \vec{\xi}_\vec{r}, \quad \textrm{where $\vec{\xi}_\vec{r}$ satisfies} \\
\left(A^T A - s^2 I_N\right) \vec{\xi}_\vec{r} = \left( I_N - \vec{r} \vec{r}^T \right) \overline{\vec{r}}, \quad \vec{r}^T \vec{\xi}_\vec{r} = 0.
\label{eq: xi r}
$$



Owning to the third equation of $\eqref{eq: adjoint equation}$, $c_\vec{l}$ and $c_\vec{r}$ satisfy the additional condition


$$
-2s c_\vec{l} - 2s c_\vec{r} - \eta_s = \overline{s}.
\label{eq: cl and cr}
$$


As the final step, the desired adjoint of a certain parameter $\theta_\mu$ reads


$$
\begin{align}
\overline{\theta_\mu} &= -\vec{\eta}^T \frac{\partial f}{\partial \theta_\mu} \nonumber \\
&\rightarrow 
-\vec{\eta}_\vec{l}^T \frac{\partial A}{\partial \theta_\mu} A^T \vec{l}
-\vec{\eta}_\vec{l}^T A \frac{\partial A^T}{\partial \theta_\mu} \vec{l}
-\vec{\eta}_\vec{r}^T \frac{\partial A^T}{\partial \theta_\mu} A \vec{r}
-\vec{\eta}_\vec{r}^T A^T \frac{\partial A}{\partial \theta_\mu} \vec{r}
-\eta_s \vec{l}^T \frac{\partial A}{\partial \theta_\mu} \vec{r}.
\end{align}
$$


One can strip the parameter $\vec{\theta}$ out of the primitive by taking account of the fact that $\overline{\theta_\mu} = \textrm{Tr}\left(\overline{A}^T \frac{\partial A}{\partial \theta_\mu}\right)$. We thus finally obtain the adjoint relation $\overline{A} = \overline{A}(\overline{\vec{l}}, \overline{\vec{r}}, \overline{s})$ as follows:


$$
\begin{align}
\overline{A} &= 
-\vec{\eta}_\vec{l} \vec{l}^T A - \vec{l} \vec{\eta}_\vec{l}^T A
-A \vec{r} \vec{\eta}_\vec{r}^T - A \vec{\eta}_\vec{r} \vec{r}^T
-\eta_s \vec{l} \vec{r}^T \nonumber \\
&= \overline{s} \vec{l} \vec{r}^T
-\vec{\xi}_\vec{l} \vec{l}^T A - \vec{l} \vec{\xi}_\vec{l}^T A
-A \vec{r} \vec{\xi}_\vec{r}^T - A \vec{\xi}_\vec{r} \vec{r}^T \nonumber \\
&= \overline{s} \vec{l} \vec{r}^T
-s \vec{\xi}_\vec{l} \vec{r}^T - \vec{l} \vec{\xi}_\vec{l}^T A
-s \vec{l} \vec{\xi}_\vec{r}^T - A \vec{\xi}_\vec{r} \vec{r}^T.
\label{eq: adjoint relation}
\end{align}
$$


where we have used $\eqref{eq: additional gauge condition}$ and $\eqref{eq: cl and cr}$ in the last two steps. Eqs. $\eqref{eq: xi l}$, $\eqref{eq: xi r}$ and $\eqref{eq: adjoint relation}$ together characterize the automatic differentiation of truncated SVD in a full-spectrum-free form.

## Relation with the approach based on full SVD

---

In this section, we will derive the back-propagation of truncated SVD by wrapping within it the corresponding process of full SVD. The relevant notations are the same as that in the first section. The adjoint relation of the full SVD process $\overline{A} = \overline{A}(\overline{U}, \overline{V}, \overline{S})$ is standard and can be easily obtained. For details, see the references listed in the last section. The final result is:


$$
\overline{A} =
U \left( \overline{S} \circ I_k \right) V^T + 
U \left[\left(U^T \overline{U} - \overline{U}^T U\right) \circ F\right] S V^T + 
\left( I_M - U U^T\right) \overline{U} S^{-1} V^T \\ + 
U S \left[\left(V^T \overline{V} - \overline{V}^T V\right) \circ F\right] V^T + 
U S^{-1} \overline{V}^T \left( I_N - V V^T \right).
\label{eq: adjoint relation full SVD}
$$



where $F$ is a $k \times k$ anti-symmetric matrix with off-diagonal elements $F_{ij} = (s_j^2 - s_i^2)^{-1}$ and $\circ$ denotes the Hadamard element-wise product.

Without loss of generality, let the desired singular value $s$ and singular vectors $\vec{l}$, $\vec{r}$ be the "first" one of the full spectrum. In other words, we have


$$
U = \begin{pmatrix}
				\vert & \vert & & \vert \\
				\vec{l} & \vec{l}_2 & \cdots & \vec{l}_k \\
				\vert & \vert & & \vert
			\end{pmatrix}, 
V = \begin{pmatrix}
				\vert & \vert & & \vert \\
				\vec{r} & \vec{r}_2 & \cdots & \vec{r}_k \\
				\vert & \vert & & \vert
			\end{pmatrix},
S = \begin{pmatrix}
		s \\
		& s_2 \\
		& & \ddots \\
		& & & s_k.
	\end{pmatrix}.
\label{eq: U V S l r s}
$$


The procedure of wrapping the process of full SVD within the truncated one then implies that the adjoints of $U$, $V$ and $S$ should take the following form:


$$
\overline{U} = 
\begin{pmatrix}
	\begin{array}{c|}
		\vert \\ \overline{\vec{l}} \\ \vert
	\end{array} & 
		\begin{array}{ccc} \\ & \Huge{0} & \\ &
	\end{array}
\end{pmatrix}, 
\overline{V} = 
\begin{pmatrix}
	\begin{array}{c|}
		\vert \\ \overline{\vec{r}} \\ \vert
	\end{array} & 
		\begin{array}{ccc} \\ & \Huge{0} & \\ &
	\end{array}
\end{pmatrix}, 
\overline{S} \circ I_k = 
\begin{pmatrix}
	\overline{s} \\
	& 0 \\
	& & \ddots \\
	& & & 0
\end{pmatrix}.
\label{eq: adjoints U V S l r s}
$$


Substituting $\eqref{eq: U V S l r s}$ and $\eqref{eq: adjoints U V S l r s}$ into Eq. $\eqref{eq: adjoint relation full SVD}$ yields the following results for each term:


$$
U \left( \overline{S} \circ I_k \right) V^T = \overline{s} \vec{l} \vec{r}^T, \\
U \left[\left(U^T \overline{U} - \overline{U}^T U\right) \circ F\right] S V^T = -
\sum_{i=2}^k \frac{1}{s_i^2 - s^2} \vec{l}_i^T \overline{\vec{l}} \cdot s \cdot \vec{l}_i \vec{r}^T -
\sum_{i=2}^k \frac{1}{s_i^2 - s^2} \vec{l}_i^T \overline{\vec{l}} \cdot s_i \cdot \vec{l} \vec{r}_i^T, \\
U S \left[\left(V^T \overline{V} - \overline{V}^T V\right) \circ F\right] V^T = -
\sum_{i=2}^k \frac{1}{s_i^2 - s^2} \vec{r}_i^T \overline{\vec{r}} \cdot s_i \cdot \vec{l}_i \vec{r}^T -
\sum_{i=2}^k \frac{1}{s_i^2 - s^2} \vec{r}_i^T \overline{\vec{r}} \cdot s \cdot \vec{l} \vec{r}_i^T, \\
\left( I_M - U U^T\right) \overline{U} S^{-1} V^T = 
\frac{1}{s} \left( I_M - U U^T\right) \overline{\vec{l}} \vec{r}^T, \\
U S^{-1} \overline{V}^T \left( I_N - V V^T \right) = 
\frac{1}{s} \vec{l} \overline{\vec{r}}^T \left( I_N - V V^T \right).
\label{eq: adjoint relation explicitly dependent form}
$$


To make clear the relation between Eq. $\eqref{eq: adjoint relation explicitly dependent form}$ and the earlier result $\eqref{eq: adjoint relation}$, we should expand the vector $\vec{\xi}\_\vec{l}$/$\vec{\xi}\_\vec{r}$ characterized by Eq. $\eqref{eq: xi l}$/$\eqref{eq: xi r}$ in the complete basis $(\vec{l}, \vec{l}\_2, \cdots, \vec{l}\_M)$/$(\vec{r}, \vec{r}\_2, \cdots, \vec{r}\_N)$, respectively. Making use of the original relations $\eqref{eq: leigenvector}$-$\eqref{eq: normalization}$, it is easy to obtain the expansion coefficients for $\vec{\xi}_\vec{l}$ as follows:


$$
\vec{l}_i^T \vec{\xi}_\vec{l} = 
\begin{cases}
	0. & i = 1. \\
	\frac{1}{s_i^2 - s^2} \vec{l}_i^T \overline{\vec{l}}. & i = 2, \cdots, k. \\
	-\frac{1}{s^2} \vec{l}_i^T \overline{\vec{l}}. & i = k+1, \cdots, M.
\end{cases}
$$


The same result also holds for $\vec{r}\_i^T \vec{\xi}\_\vec{r}$. The expansion expression of  $\vec{\xi}\_\vec{l}$  and $\vec{\xi}\_\vec{r}$ are thus as follows:


$$
\vec{\xi}_\vec{l} = 
\sum_{i=2}^k \frac{1}{s_i^2 - s^2} \vec{l}_i^T \overline{\vec{l}} \cdot \vec{l}_i - 
\frac{1}{s^2} \left(I_M - U U^T\right) \overline{\vec{l}}.
\label{eq: xi l expansion}
$$

$$
\vec{\xi}_\vec{r} = 
\sum_{i=2}^k \frac{1}{s_i^2 - s^2} \vec{r}_i^T \overline{\vec{r}} \cdot \vec{r}_i - 
\frac{1}{s^2} \left(I_N - V V^T\right) \overline{\vec{r}}.
$$



where we have used the completeness relations $U U^T + \sum_{i=k+1}^M \vec{l}\_i \vec{l}\_i^T = I_M$ and $V V^T + \sum_{i=k+1}^N \vec{r}\_i \vec{r}\_i^T = I_N$ to eliminate the eigenvectors with eigenvalue zero. Furthermore, by making use of Eqs. $\eqref{eq: from l to r}$ and $\eqref{eq: from r to l}$, we have


$$
A^T \vec{\xi}_\vec{l} = 
\sum_{i=2}^k \frac{1}{s_i^2 - s^2} \vec{l}_i^T \overline{\vec{l}} \cdot s_i \cdot \vec{r}_i.
$$

$$
A \vec{\xi}_\vec{r} = 
\sum_{i=2}^k \frac{1}{s_i^2 - s^2} \vec{r}_i^T \overline{\vec{r}} \cdot s_i \cdot \vec{l}_i.
\label{eq: A xi r expansion}
$$



Motivated by the expansion expressions $\eqref{eq: xi l expansion}$-$\eqref{eq: A xi r expansion}$, one can readily recognize the relevant pieces and convert the original results $\eqref{eq: adjoint relation explicitly dependent form}$ into a more compact form:


$$
\overline{A} = 
\overline{s} \vec{l} \vec{r}^T
- s \vec{\xi}_\vec{l} \vec{r}^T
- s \vec{l} \vec{\xi}_\vec{r}^T
- \vec{l} \vec{\xi}_\vec{l}^T A
- A \vec{\xi}_\vec{r} \vec{r}^T.
\label{eq: adjoint relation again}
$$


Eq. $\eqref{eq: adjoint relation again}$ is identically the same as the result $\eqref{eq: adjoint relation}$ obtained through the adjoint method. 

Note that the original adjoint relation $\eqref{eq: adjoint relation full SVD}$ of full SVD is pretty lengthy. However, when it is used to derive the back-propagation of truncated SVD, this complication is considerably reduced by the vectors $\vec{\xi}\_\vec{l}$ and $\vec{\xi}\_\vec{r}$. What happened is that these two vectors take care of the components within the eigen-subspace of both nonzero and zero eigenvalues at the same time.

## References

---

There are many texts and lecture notes covering the basics of SVD. For example, 

- [https://graphics.stanford.edu/courses/cs205a-13-fall/assets/notes/chapter6.pdf](https://graphics.stanford.edu/courses/cs205a-13-fall/assets/notes/chapter6.pdf)

The derivation of the back-propagation (i.e., adjoint relation) of full SVD can be found in, say

- [https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf](https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf)

- [https://j-towns.github.io/papers/svd-derivative.pdf](https://j-towns.github.io/papers/svd-derivative.pdf)
