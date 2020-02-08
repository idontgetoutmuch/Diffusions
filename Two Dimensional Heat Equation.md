
# A Biological Example

Suppose we are observing exponential growth $\dot{y} = \theta y$ but we don't know $\theta$ and wish to estimate it. We could assume $\theta \sim {\cal{N}}(\mu, \sigma^2)$ and use something like Markov Chain Monte Carlo or Hamiltonian Monte Carlo and any observations to infer $\mu$ and $\sigma$. However, we might want to model that the further we go into the future, the less we know about $\theta$. We can write our system as as

$$
\begin{aligned}
\mathrm{d}y & = \theta y\mathrm{d}t \\
\mathrm{d}\theta & = \sigma\mathrm{d}W_t
\end{aligned}
$$

where $W_t$ is Brownian Motion.

# Fokker-Planck

$$
d \mathbf{X}_{t}=\boldsymbol{\mu}\left(\mathbf{X}_{t}, t\right) d t+\boldsymbol{\sigma}\left(\mathbf{X}_{t}, t\right) d \mathbf{W}_{t}
$$

$$
\frac{\partial}{\partial t} p(t, \mathbf{x})+\sum_{k=1}^{n} \frac{\partial}{\partial x_{k}}\left({\mu}_{k}(t, \mathbf{x}) p(t, \mathbf{x})\right)=\frac{1}{2} \sum_{j=1, k=1}^{n} \frac{\partial^{2}}{\partial x_{j} \partial x_{k}}\left[\left(\sigma(t, \mathbf{x}) \sigma^{T}(t, \mathbf{x})\right)_{j k} p(t, \mathbf{x})\right]
$$

For our particular system we have

$$
\frac{\partial}{\partial t} p(t, y, \theta)+\frac{\partial}{\partial y}\left({\mu}_{1}(t, y, \theta) p(t, y, \theta)\right)+\frac{\partial}{\partial \theta}\left({\mu}_{2}(t, y, \theta) p(t, y, \theta)\right)=\frac{1}{2}\left[\sigma_{y}^{2} \frac{\partial^{2}}{\partial y^{2}} p(t, y, \theta)+\sigma_{\theta}^{2} \frac{\partial^{2}}{\partial \theta^{2}} p(t, y, \theta)\right]
$$

And since $\mu_1 = \theta y$, $\mu_2 = 0$ and $\sigma_y = 0$ this further simplifies to

$$
\frac{\partial}{\partial t} p(t, y, \theta)+\frac{\partial}{\partial y}(\theta y p(t, y, \theta))=\sigma_{\theta}^{2} \frac{\partial^{2}}{\partial \theta^{2}} p(t, y, \theta)
$$

We can note two things:

* This is an advection / diffusion equation with two spatial variables ($y$ and $\theta$).
* If $\sigma_\theta = 0$ then this is a transport (advection?) equation.

$$
\frac{\partial}{\partial t} p(t, y, \theta)+\frac{\partial}{\partial y}(\theta y p(t, y, \theta))=0
$$

Notice that there is nothing stochastic about the biology but we
express our uncertainty about the parameter by making it a
time-varying stochastic variable which says the further we go into the
future the less certain we are about it.

We are going to turn this into a Fokker-Planck equation which we can
then solve using e.g. the method of lines. But before turning to
Fokker-Planck, let's show that we can indeed solve a diffusion
equation using the method of lines.

We want to solve the heat equation

$$
\frac{\partial u}{\partial t} = k_x\frac{\partial^2 u}{\partial x^2} + k_y\frac{\partial^2 u}{\partial x^2}
$$

The spatial derivatives are computed using second-order centered differences, with the data distributed over $n_x \times n_y$ points on a uniform spatial grid.

$$
u_{i\,j}(t) \triangleq u\left(t, x_{i}, y_{j}\right), \quad x_{i} \triangleq i \Delta x, \quad 0 \leq i \leq n_x+1, \quad  y_{j} \triangleq j \Delta y, \quad 0 \leq j \leq n_y+1
$$

$$
\begin{align}
u_{x x} &= \frac{u_{i+1\,j}-2 u_{i\,j}+u_{i-1\,j}}{\Delta x^{2}} \\
u_{y y} &= \frac{u_{i\,j+1}-2 u_{i\,j}+u_{i\,j-1}}{\Delta y^{2}}
\end{align}
$$

$$
\dot{u}_{i\, j} = \frac{k_x}{(\Delta x)^2}({u_{i+1\,j}-2 u_{i\,j}+u_{i-1\,j}})
                + \frac{k_y}{(\Delta y)^2}({u_{i\,j+1}-2 u_{i\,j}+u_{i\,j-1}})
$$



$$
\dot{u}_{i\, j} = \sum_{k=0}^{n+1}\sum_{l=0}^{n+1}A_{i\,j\,k\,l} u_{k\,l}
$$

$$
\begin{align}
A_{0\,j\,k\,l} &= 0 \\
A_{i\,j\,i-1\,j} &= 1 \\
A_{i\,j\,i\,j} &= -2 \\
A_{i\,j\,i+1\,j} &= 1 \\
A_{n+1\,j\,k\,l} &= 0 \\
\end{align}
$$

We could try using [Naperian functors and APL-like programming in Haskell](https://www.cs.ox.ac.uk/people/jeremy.gibbons/publications/aplicative.pdf) via this [library](http://hackage.haskell.org/package/Naperian). But the performance is terrible (or it could be that the author's implementation was terrible). Moreover, applied mathematicans tend to think of everything as matrices and vectors. But flattening the above tensor operation into a matrix operation is not entirely trivial. Although the Haskell Ecosystem's support for symbolic mathematics is very rudimentary, we can use what there is to convince ourselves that we haven't made too many errors in the transcription.


```haskell
{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}

import Data.Maybe
import Data.Number.Symbolic
import qualified Data.Number.Symbolic as Sym
import Data.Proxy

import qualified Naperian as N
import qualified Data.Foldable as F
import           Control.Applicative ( liftA2 )
import qualified GHC.TypeLits as M
import           Data.Functor
import           Data.List.Split

import           Numeric.Sundials.ARKode.ODE
import           Numeric.LinearAlgebra
```

Heat conductivity coefficients:


```haskell
kx, ky :: Floating a => a
kx = 0.5
ky = 0.75
```


```haskell
-- spatial mesh size
nx, ny :: Int
nx = 30
ny = 60
```


```haskell
-- x mesh spacing
-- y mesh spacing
dx :: Floating a => a
dx = 1 / (fromIntegral nx - 1)

dy :: Floating a => a
dy = 1 / (fromIntegral ny - 1)

c1, c2 :: Floating a => a
c1 = kx/dx/dx
c2 = ky/dy/dy

cc4' :: forall b m n . (M.KnownNat m, M.KnownNat n, Num b) =>
        N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] b
cc4' = N.Prism $ N.Prism $ N.Prism $ N.Prism $ N.Scalar $
      N.viota @m <&> (\(N.Fin x) ->
      N.viota @n <&> (\(N.Fin w) ->
      N.viota @m <&> (\(N.Fin v) ->
      N.viota @n <&> (\(N.Fin u) ->
      (f m n x w v u)))))
        where
          m = fromIntegral $ M.natVal (undefined :: Proxy m)
          n = fromIntegral $ M.natVal (undefined :: Proxy n)
          f m n i j k l | i == 0               = 0
                        | j == 0               = 0
                        | i == n - 1           = 0
                        | j == m - 1           = 0
                        | k == i - 1 && l == j = 1
                        | k == i     && l == j = -2
                        | k == i + 1 && l == j = 1
                        | otherwise            = 0
```


```haskell
cc5' :: forall a m n . (M.KnownNat m, M.KnownNat n, Floating a) =>
        N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] a
cc5' = N.binary (*) (N.Scalar c2) cc4'

cc5Sym' :: forall a m n . (M.KnownNat m, M.KnownNat n, Floating a, Eq a) =>
          N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] (Sym a)
cc5Sym' = N.binary (*) (N.Scalar $ var "c2") cc4'
```


```haskell
yy4' :: forall b m n . (M.KnownNat m, M.KnownNat n, Num b) =>
        N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] b
yy4' = N.Prism $ N.Prism $ N.Prism $ N.Prism $ N.Scalar $
      N.viota @m <&> (\(N.Fin x) ->
      N.viota @n <&> (\(N.Fin w) ->
      N.viota @m <&> (\(N.Fin v) ->
      N.viota @n <&> (\(N.Fin u) ->
      (f m n x w v u)))))
        where
          m = fromIntegral $ M.natVal (undefined :: Proxy m)
          n = fromIntegral $ M.natVal (undefined :: Proxy n)
          f :: Int -> Int -> Int -> Int -> Int -> Int -> b
          f m n i j k l | i == 0                   = 0
                        | j == 0                   = 0
                        | i == n - 1               = 0
                        | j == m - 1               = 0
                        | k == i     && l == j - 1 = 1
                        | k == i     && l == j     = -2
                        | k == i     && l == j + 1 = 1
                        | otherwise                = 0

yy5' :: forall a m n . (M.KnownNat m, M.KnownNat n, Floating a) =>
        N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] a
yy5' = N.binary (*) (N.Scalar c1) yy4'

yy5Sym' :: forall a m n . (M.KnownNat m, M.KnownNat n, Floating a, Eq a) =>
           N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] (Sym a)
yy5Sym' = N.binary (*) (N.Scalar $ var "c1") yy4'
```


```haskell
ccSym5 = cc5Sym' @Double @4 @5
yy5Sym = yy5Sym' @Double @4 @5
```


```haskell
ccSym5
```


    <<<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>>,<<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,c2,0.0,0.0,0.0>,<0.0,(-2.0)*c2,0.0,0.0,0.0>,<0.0,c2,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,c2,0.0,0.0>,<0.0,0.0,(-2.0)*c2,0.0,0.0>,<0.0,0.0,c2,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,c2>,<0.0,0.0,0.0,0.0,(-2.0)*c2>,<0.0,0.0,0.0,0.0,c2>,<0.0,0.0,0.0,0.0,0.0>>>,<<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,c2,0.0,0.0,0.0>,<0.0,(-2.0)*c2,0.0,0.0,0.0>,<0.0,c2,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,c2,0.0,0.0>,<0.0,0.0,(-2.0)*c2,0.0,0.0>,<0.0,0.0,c2,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,c2>,<0.0,0.0,0.0,0.0,(-2.0)*c2>,<0.0,0.0,0.0,0.0,c2>>>,<<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,c2,0.0,0.0,0.0>,<0.0,(-2.0)*c2,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,c2,0.0,0.0>,<0.0,0.0,(-2.0)*c2,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,c2>,<0.0,0.0,0.0,0.0,(-2.0)*c2>>>>



```haskell
yy5Sym
```


    <<<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>>,<<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<c1,(-2.0)*c1,c1,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,c1,(-2.0)*c1,c1,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,c1,(-2.0)*c1>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>>,<<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<c1,(-2.0)*c1,c1,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,c1,(-2.0)*c1,c1,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,c1,(-2.0)*c1>,<0.0,0.0,0.0,0.0,0.0>>>,<<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<c1,(-2.0)*c1,c1,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,c1,(-2.0)*c1,c1,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>>,<<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,0.0,0.0>,<0.0,0.0,0.0,c1,(-2.0)*c1>>>>



```haskell
fmap (N.elements . N.Prism . N.Prism . N.Scalar) $ N.elements $ N.crystal $ N.crystal $ N.binary (+) cc5Sym yy5Sym
```


    [[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,c2,0.0,0.0,0.0,c1,(-2.0)*c2+(-2.0)*c1,c1,0.0,0.0,0.0,c2,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,c2,0.0,0.0,0.0,c1,(-2.0)*c2+(-2.0)*c1,c1,0.0,0.0,0.0,c2,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,c2,0.0,0.0,0.0,0.0,(-2.0)*c2,0.0,0.0,0.0,0.0,c2,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,c1,(-2.0)*c1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,c2,0.0,0.0,0.0,c1,(-2.0)*c2+(-2.0)*c1,c1,0.0,0.0,0.0,c2,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,c2,0.0,0.0,0.0,c1,(-2.0)*c2+(-2.0)*c1,c1,0.0,0.0,0.0,c2,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,c2,0.0,0.0,0.0,0.0,(-2.0)*c2,0.0,0.0,0.0,0.0,c2,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,c1,(-2.0)*c1,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,c1,(-2.0)*c1,c1,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,c1,(-2.0)*c1,c1,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,c1,(-2.0)*c1]]


Fokker-Planck
=============

$$
\frac{\partial}{\partial t} p(t, \mathbf{x})+\sum_{k=1}^{n} \frac{\partial}{\partial x_{k}}\left(g_{k}(t, \mathbf{x}) p(t, \mathbf{x})\right)=\frac{1}{2} \sum_{j=1, k=1}^{n} \frac{\partial^{2}}{\partial x_{j} \partial x_{k}}\left[\left(\sigma(t, \mathbf{x}) \sigma^{T}(t, \mathbf{x})\right)_{j k} p(t, \mathbf{x})\right]
$$

$$
\frac{\partial}{\partial t} p(t, y, \theta)+\frac{\partial}{\partial y}(f(t, y, \theta ; k) p(t, y, \theta))=\frac{1}{2}\left[\sigma_{y}^{2} \frac{\partial^{2}}{\partial y^{2}} p(t, y, \theta)+\sigma_{\theta}^{2} \frac{\partial^{2}}{\partial \theta^{2}} p(t, y, \theta)\right]
$$

$$
\frac{\partial}{\partial t} p(t, y, \theta)+\frac{\partial}{\partial y}(f(t, y, \theta ; k) p(t, y, \theta))=0
$$

Since

$$
\dot{y}=\theta y\left(1-\frac{y}{k}\right)
$$

we have

$$
f(t, y, \theta ; k)=\theta y\left(1-\frac{y}{k}\right)
$$

$$
\frac{\partial p}{\partial t} +\frac{\partial}{\partial y}\bigg(\theta y\bigg(1 - \frac{y}{k}\bigg) p\bigg)=0
$$

$$
\frac{\partial p}{\partial t} + p\frac{\partial}{\partial y}\bigg(\theta y\bigg(1 - \frac{y}{k}\bigg) \bigg) + \theta y\bigg(1 - \frac{y}{k}\bigg)\frac{\partial p}{\partial y} = 0
$$

$$
\frac{\partial p}{\partial t} + p\theta\bigg(1 - \frac{y}{k} - \frac{y}{k}\bigg) + \theta y\bigg(1 - \frac{y}{k}\bigg)\frac{\partial p}{\partial y} = 0
$$

$$
\frac{\partial p}{\partial t} + \theta y\bigg(1 - \frac{y}{k}\bigg)\frac{\partial p}{\partial y} = - p\theta\bigg(1 - \frac{2y}{k}\bigg)
$$

We can solve the transport PDE with initial condition
$$
\left\{\begin{array}{l}{u_{t}+a u_{x}=0} \\ {u(x, 0)=\phi(x)}\end{array}\right.
$$

using the Method of Characteristics with $a = 2$ and $\phi (x) =
e^{-x^2}$ to give the solutuion illustrated

![](diagrams/transport.png)

$$
\begin{aligned}
\frac{d S}{d t} &=-\delta S(t) I(t) \\
\frac{d I}{d t} &=\delta S(t) I(t)-\gamma I(t) \\
\frac{d R}{d t} &=\quad \gamma I(t)
\end{aligned}
$$

$$
\begin{aligned}
s^{\prime}(t) &=-\beta s(t) i(t)-\mu s(t)+\mu \\
i^{\prime}(t)    &= \beta s(t) i(t)-\mu i(t)
\end{aligned}
$$


$$
\begin{aligned}
0 &=-\beta s_{\infty} i_{\infty}-\mu s_{\infty}+\mu \\
0 &= \beta s_{\infty} i_{\infty}-\mu i_{\infty}
\end{aligned}
$$

$$
\begin{aligned}
0 &= -\mu i_{\infty} -\mu s_{\infty}+\mu
\end{aligned}
$$

$$
\begin{aligned}
s_{\infty} + i_{\infty} = 1
\end{aligned}
$$

$$
\begin{aligned}
\beta (1 - i_{\infty})i_{\infty} - \mu i_{\infty} &= 0 \\
(1 - i_{\infty}) i_{\infty} - \frac{\mu}{\beta} i_{\infty} &= 0
\end{aligned}
$$

$$
\begin{aligned}
i_{\infty} &= 1 - \frac{\mu}{\beta} \\
s_{\infty} &=     \frac{\mu}{\beta}
\end{aligned}
$$

$$
i(t)=\frac{\lambda}{\beta+\lambda\left(\frac{\lambda-i_{0} \beta}{\lambda i_{0} e} \frac{\beta-i_{0} \beta}{\mu}\right) e^{-\lambda t+\frac{\beta\left(s_{0}+i_{0}-1\right)}{\mu}}}
$$

If we let $t \rightarrow \infty$ then we obtain $i_{\infty} =
\frac{\lambda}{\beta}$.

$$
\left[ \begin{array}{c}{y_{i}} \\ {\theta_{i}}\end{array}\right]=\left[ \begin{array}{c}{\frac{k y_{i-1} \exp \theta_{i-1}\left(t_{i}-t_{i-1}\right)}{k+y_{i-1}\left(\exp \theta_{i-1}\left(t_{i}-t_{i-1}\right)-1\right)}} \\ {\theta_{i-1}}\end{array}\right]+\psi_{i-1}
$$

We just need to apply the MoC to the PDE for the probability distribution.

$$
\left[ \begin{array}{c}{y_{i}} \\ {\log \theta_{i}}\end{array}\right]=\left[ \begin{array}{c}{\frac{k y_{i-1} \exp \theta_{i-1}\left(t_{i}-t_{i-1}\right)}{k+y_{i-1}\left(\exp \theta_{i-1}\left(t_{i}-t_{i-1}\right)-1\right)}} \\ {\log \theta_{i-1}}\end{array}\right]+\psi_{i-1}
$$

$$
\left[ \begin{array}{c}{z_{i}}\end{array}\right]=\left[ \begin{array}{ll}{1} & {0}\end{array}\right] \left[ \begin{array}{l}{y_{i}} \\ {\theta_{i}}\end{array}\right]+\nu_{i}
$$

$$
\psi_{i} \sim \mathcal{N}(0, Q)
$$

$$
v_{i} \sim \mathcal{N}(0, R)
$$

$$
f(t, y, \theta ; k)=\frac{k y_{0} \exp \theta t}{k+y_{0}(\exp \theta t-1)}
$$

$$
\dot{y}=\theta y\left(1-\frac{y}{k}\right)
$$

$$
y=\frac{k y_{0} \exp \theta t}{k+y_{0}(\exp \theta t-1)}
$$


Let us solve the heat equation over the unit square to some arbitrary
point in the future.

$$
\frac{\partial u}{\partial t}=k_{x} \frac{\partial^{2} u}{\partial x^{2}}+k_{y} \frac{\partial^{2} u}{\partial y^{2}}+h
$$

with initial condition $u(0, x, y) = 0$ and stationary boundary conditions

$$
\frac{\partial u}{\partial t}(t, 0, y)=\frac{\partial u}{\partial t}(t, 1, y)=\frac{\partial u}{\partial t}(t, x, 0)=\frac{\partial u}{\partial t}(t, x, 1)=0
$$

and a periodic heat source

$$
h(x, y)=\sin (\pi x) \sin (2 \pi y)
$$

This has analytic solution

$$
u(t, x, y)=\frac{1-e^{-\left(k_{x}+4 k_{y}\right) \pi^{2} t}}{\left(k_{x}+4 k_{y}\right) \pi^{2}} \sin (\pi x) \sin (2 \pi y)
$$

$$
u_{i\,j}(t) \triangleq u\left(t, x_{i}, y_{j}\right), \quad x_{i} \triangleq i \Delta x, \quad 0 \leq j \leq n+1, \quad  y_{j} \triangleq j \Delta y
$$


$$
\begin{align}
u_{x x} &= \frac{u_{i+1\,j}-2 u_{i\,j}+u_{i-1\,j}}{\Delta x^{2}} \\
u_{y y} &= \frac{u_{i\,j+1}-2 u_{i\,j}+u_{i\,j-1}}{\Delta y^{2}}
\end{align}
$$

$$
\dot{u}_{i\, j} = \frac{k_x}{(\Delta x)^2}({u_{i+1\,j}-2 u_{i\,j}+u_{i-1\,j}})
                + \frac{k_y}{(\Delta y)^2}({u_{i\,j+1}-2 u_{i\,j}+u_{i\,j-1}})
$$

$$
\dot{u}_{i\, j} = \sum_{k=0}^{n+1}\sum_{l=0}^{n+1}A_{i\,j\,k\,l} u_{k\,l}
$$

$$
\begin{align}
A_{0\,j\,k\,l} &= 0 \\
A_{i\,j\,i-1\,j} &= 1 \\
A_{i\,j\,i\,j} &= -2 \\
A_{i\,j\,i+1\,j} &= 1 \\
A_{n+1\,j\,k\,l} &= 0 \\
\end{align}
$$


```haskell
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE OverloadedLists       #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE GADTs               #-}
```


```haskell
import Data.Maybe
import Data.Number.Symbolic
import Data.Proxy

import qualified Naperian as N
import qualified Data.Foldable as F
import           Control.Applicative ( liftA2 )
import qualified GHC.TypeLits as M

import           Numeric.Sundials.ARKode.ODE
import           Numeric.LinearAlgebra
```


```haskell
x1, a, x2 :: Double
x1 = 0
a = 1.0
x2 = a
```


```haskell
y1, y2 :: Double
y1 = 0.0
y2 = 1.0
```


```haskell
bigT :: Double
bigT = 1000.0
```


```haskell
n :: Int
n = 2
```


```haskell
dx :: Double
dx = a / (fromIntegral n + 1)
```


```haskell
dy :: Double
dy = a / (fromIntegral n + 1)
```


```haskell
beta, s :: Double
beta = 1.0e-5
s = beta / (dx * dx)
```

Heat conductivity coefficients


```haskell
kx, ky :: Double
kx = 0.5
ky = 0.75
```


```haskell
c1, c2 :: Double
c1 = kx/dx/dx
c2 = ky/dy/dy
```

We want to turn this into a matrix equation so that we can use `hmatrix-sundials`


```haskell
bigAA1 :: Matrix Double
bigAA1 = assoc (n * n, n * n) 0.0 [((i, j), f (i, j)) | i <- [0 .. n * n - 1]
                                                      , j <- [i - n, i,  i + n]
                                                      , j `elem` [0 .. n * n -1]]
  where
    f (i, j) | i     == j = (-2.0) * c1
             | i - n == j = 1.0    * c1
             | i + n == j = 1.0    * c1
             | otherwise = error $ show (i, j)

bigAA2 :: Matrix Double
bigAA2 = diagBlock (replicate n bigA)
  where
    bigA :: Matrix Double
    bigA = assoc (n, n) 0.0 [((i, j), f (i, j)) | i <- [0 .. n - 1]
                                                , j <- [i-1..i+1]
                                                , j `elem` [0..n-1]]
      where
        f (i, j) | i     == j = (-2.0) * c2
                 | i - 1 == j = 1.0    * c2
                 | i + 1 == j = 1.0    * c2

bigAA :: Matrix Double
bigAA = bigAA1 + bigAA2
```


```haskell
bigAA1
```


    (4><4)
     [ -9.0,  0.0,  4.5,  0.0
     ,  0.0, -9.0,  0.0,  4.5
     ,  4.5,  0.0, -9.0,  0.0
     ,  0.0,  4.5,  0.0, -9.0 ]



```haskell
bigAA2
```


    (4><4)
     [ -13.5,  6.75,   0.0,   0.0
     ,  6.75, -13.5,   0.0,   0.0
     ,   0.0,   0.0, -13.5,  6.75
     ,   0.0,   0.0,  6.75, -13.5 ]


$$
h(x, y)=\sin (\pi x) \sin (2 \pi y)
$$


```haskell
n
```


    2



```haskell
bigZZ1 :: Matrix Double
bigZZ1 = assoc (m * m, m * m) 0.0 [((i, j), f (i, j)) | i <- [0 .. m * m - 1]
                                                      , j <- [0 .. m * m - 1]]
  where
    m = n + 2
    f (i, j) | i     == 0     = 0.0
             | j     == 0     = 0.0
             | i     == j     = (-2.0) * c1
             | i - n == j     = 1.0    * c1
             | i + n == j     = 1.0    * c1
             | i     == n + 1 = 0.0
             | j     == n + 1 = 0.0
             | otherwise      = 0.0

```


```haskell
bigZZ1
```


    (16><16)
     [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
     , 0.0, -9.0,  0.0,  4.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
     , 0.0,  0.0, -9.0,  0.0,  4.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
     , 0.0,  4.5,  0.0, -9.0,  0.0,  4.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
     , 0.0,  0.0,  4.5,  0.0, -9.0,  0.0,  4.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
     , 0.0,  0.0,  0.0,  4.5,  0.0, -9.0,  0.0,  4.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
     , 0.0,  0.0,  0.0,  0.0,  4.5,  0.0, -9.0,  0.0,  4.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
     , 0.0,  0.0,  0.0,  0.0,  0.0,  4.5,  0.0, -9.0,  0.0,  4.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0
     , 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.5,  0.0, -9.0,  0.0,  4.5,  0.0,  0.0,  0.0,  0.0,  0.0
     , 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.5,  0.0, -9.0,  0.0,  4.5,  0.0,  0.0,  0.0,  0.0
     , 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.5,  0.0, -9.0,  0.0,  4.5,  0.0,  0.0,  0.0
     , 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.5,  0.0, -9.0,  0.0,  4.5,  0.0,  0.0
     , 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.5,  0.0, -9.0,  0.0,  4.5,  0.0
     , 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.5,  0.0, -9.0,  0.0,  4.5
     , 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.5,  0.0, -9.0,  0.0
     , 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  4.5,  0.0, -9.0 ]


$$
\dot{u}_{i\, j} = \frac{k_x}{(\Delta x)^2}({u_{i+1\,j}-2 u_{i\,j}+u_{i-1\,j}})
                + \frac{k_y}{(\Delta y)^2}({u_{i\,j+1}-2 u_{i\,j}+u_{i\,j-1}})
$$


```haskell
x :: forall m n . (M.KnownNat m, M.KnownNat n) => N.Vector n (N.Vector m (Sym Int))
x = (fromJust . N.fromList) $
    map (fromJust . N.fromList) ([[var $ (\(x,y) -> "A" ++ show x ++ "," ++ show y) (x,y) | y <- [1..m]] | x <- [1..n]] :: [[Sym Int]])
    where
      m = M.natVal (undefined :: Proxy m)
      n = M.natVal (undefined :: Proxy n)
```


```haskell
u1 :: N.Hyper '[N.Vector 3, N.Vector 2] (Sym Int)
u1 = N.Prism $ N.Prism (N.Scalar x)
```


```haskell
u1
```


    <<A1,1,A1,2,A1,3>,<A2,1,A2,2,A2,3>>



```haskell
y :: forall n . M.KnownNat n => N.Vector n (Sym Int)
y = (fromJust . N.fromList) $
    (map (var . ("v" ++) . show) [1..n ] :: [Sym Int])
    where
    n = M.natVal (undefined :: Proxy n)
```


```haskell
u2 :: N.Hyper '[N.Vector 3] (Sym Int)
u2 = N.Prism (N.Scalar y)
```


```haskell
N.innerH u1 u2
```


    <A1,1*v1+A1,2*v2+A1,3*v3,A2,1*v1+A2,2*v2+A2,3*v3>

