{-# OPTIONS_GHC -Wall            #-}

{-# LANGUAGE DataKinds           #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleContexts    #-}
{-# LANGUAGE FlexibleInstances   #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE TypeApplications    #-}
{-# LANGUAGE TypeOperators       #-}

module Heat2D (main, a1, a2) where

import Data.Number.Symbolic
import Data.Proxy

import qualified Naperian as N
import qualified GHC.TypeLits as M
import           Data.Functor
import           System.IO

import           Numeric.Sundials.ARKode.ODE
import           Numeric.LinearAlgebra


kx, ky :: Floating a => a
kx = 0.5
ky = 0.75

-- x mesh spacing
-- y mesh spacing
dx :: Floating a => a
dx = 1 / (fromIntegral nx - 1)

dy :: Floating a => a
dy = 1 / (fromIntegral ny - 1)

c1, c2 :: Floating a => a
c1 = kx/dx/dx
c2 = ky/dy/dy

preA2 :: forall b m n . (M.KnownNat m, M.KnownNat n, Num b) =>
        N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] b
preA2 = N.Prism $ N.Prism $ N.Prism $ N.Prism $ N.Scalar $
      N.viota @m <&> (\(N.Fin x) ->
      N.viota @n <&> (\(N.Fin w) ->
      N.viota @m <&> (\(N.Fin v) ->
      N.viota @n <&> (\(N.Fin u) ->
      (f m n x w v u)))))
        where
          m = fromIntegral $ M.natVal (undefined :: Proxy m)
          n = fromIntegral $ M.natVal (undefined :: Proxy n)
          f p q i j k l | i == 0               = 0
                        | j == 0               = 0
                        | i == p - 1           = 0
                        | j == q - 1           = 0
                        | k == i - 1 && l == j = 1
                        | k == i     && l == j = -2
                        | k == i + 1 && l == j = 1
                        | otherwise            = 0

a2Num :: forall a m n . (M.KnownNat m, M.KnownNat n, Floating a) =>
        N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] a
a2Num = N.binary (*) (N.Scalar c2) preA2

a2 :: forall a m n . (M.KnownNat m, M.KnownNat n, Floating a, Eq a) =>
          N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] (Sym a)
a2 = N.binary (*) (N.Scalar $ var "a") preA2

preA1 :: forall b m n . (M.KnownNat m, M.KnownNat n, Num b) =>
        N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] b
preA1 = N.Prism $ N.Prism $ N.Prism $ N.Prism $ N.Scalar $
      N.viota @m <&> (\(N.Fin x) ->
      N.viota @n <&> (\(N.Fin w) ->
      N.viota @m <&> (\(N.Fin v) ->
      N.viota @n <&> (\(N.Fin u) ->
      (f m n x w v u)))))
        where
          m = fromIntegral $ M.natVal (undefined :: Proxy m)
          n = fromIntegral $ M.natVal (undefined :: Proxy n)
          f :: Int -> Int -> Int -> Int -> Int -> Int -> b
          f p q i j k l | i == 0                   = 0
                        | j == 0                   = 0
                        | i == p - 1               = 0
                        | j == q - 1               = 0
                        | k == i     && l == j - 1 = 1
                        | k == i     && l == j     = -2
                        | k == i     && l == j + 1 = 1
                        | otherwise                = 0

a1Num :: forall a m n . (M.KnownNat m, M.KnownNat n, Floating a) =>
        N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] a
a1Num = N.binary (*) (N.Scalar c1) preA1

a1 :: forall a m n . (M.KnownNat m, M.KnownNat n, Floating a, Eq a) =>
           N.Hyper '[N.Vector n, N.Vector m, N.Vector n, N.Vector m] (Sym a)
a1 = N.binary (*) (N.Scalar $ var "b") preA1

h :: forall m n a . (M.KnownNat m, M.KnownNat n, Floating a) =>
                     N.Hyper '[N.Vector m, N.Vector n] a
h = N.Prism $ N.Prism $ N.Scalar $
     N.viota @n <&> (\(N.Fin x) ->
     N.viota @m <&> (\(N.Fin w) ->
     sin (pi * (fromIntegral w) * dx)
             * sin (2 * pi * (fromIntegral x) * dy)))

-- spatial mesh size
nx, ny :: Int
nx = 30
ny = 60

bigA :: Matrix Double
bigA = fromLists $
       fmap (N.elements . N.Prism . N.Prism . N.Scalar) $
       N.elements $ N.crystal $ N.crystal $
       N.binary (+) (a2Num @Double @60 @30) (a1Num @Double @60 @30)

b :: Vector Double
b = fromList $ N.elements (h @30 @60 @Double)

t0, tf :: Double
t0 = 0.0
tf = 0.3

bigNt :: Int
bigNt = 20

dTout :: Double
dTout = (tf - t0) / (fromIntegral bigNt)

ts :: [Double]
ts = map (dTout *) $ map fromIntegral [1..bigNt]

sol :: Matrix Double
sol = odeSolveV SDIRK_5_3_4' Nothing 1.0e-5 1.0e-10 (const bigU') (assoc (nx * ny) 0.0 [] :: Vector Double) (fromList $ ts)
  where
    bigU' bigU = bigA #> bigU + b

main :: IO ()
main = do
  h1 <- openFile "heat2e.000.txt" WriteMode
  mapM_ (hPutStrLn h1) $ map (concatMap (' ':)) $ map (map show) $ toLists sol
  hClose h1
  mapM_ (\i -> putStrLn $ show $ sqrt $ (sol!i) <.> (sol!i) / (fromIntegral nx) / (fromIntegral ny)) ([0 .. length ts - 1] :: [Int])
