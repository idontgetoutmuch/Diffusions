-- % Haskell for Numerics?
-- % Dominic Steinitz
-- % 2nd June 2017

{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE OverloadedLists     #-}
{-# LANGUAGE NumDecimals         #-}

{-# OPTIONS_GHC -Wall #-}

module Main (main) where

import           Numeric.Sundials
import           Numeric.LinearAlgebra
import           Data.Csv
import           Data.Char
import           Data.ByteString.Lazy (putStr, writeFile)
import           Prelude hiding (putStr, writeFile)
import           Control.Exception
import           Data.Coerce
import           Katip.Monadic
import           GHC.Int


bigN :: Int
bigN = 201

k :: Double
k = 0.5

deltaX :: Double
deltaX = 1.0 / (fromIntegral bigN - 1)
c1, c2 :: Double
c1 = k / deltaX / deltaX
c2 = (-2.0) * k / deltaX / deltaX

t0 :: Double
t0 = 0.0

tf :: Double
tf =1.0

bigNt :: Int
bigNt = 10

rtol :: Double
rtol = 1.0e-6

atol :: Double
atol = 1.0e-10

bigA :: Matrix Double
bigA = assoc (bigN, bigN) 0.0 [ ((i, j), f (i, j)) | i <- [0 .. bigN - 1]
                                                   , j <- [0 .. bigN - 1]
                        ]
 where
   f (i, j) | i     == 0        = 0.0    -- left boundary condition
            | i     == bigN - 1 = 0.0    -- right boundary condition
            | i     == j        = c2
            | i - 1 == j        = c1
            | i + 1 == j        = c1
            | otherwise         = 0.0

b :: Vector Double
b = assoc bigN 0.0 [ (iSource, 0.01 / deltaX) ]
  where
    iSource = bigN `div` 2

bigU0 :: Vector Double
bigU0 = assoc bigN 0.0 []

deltaT :: Double
deltaT = (tf - t0) / (fromIntegral bigNt)

sol :: IO (Matrix Double)
sol = do
  x <- runNoLoggingT $ solve (defaultOpts SDIRK_5_3_4) heat1D
  case x of
    Left e  -> error $ show e
    Right y -> return (solutionMatrix y)

heat1D :: OdeProblem
heat1D = emptyOdeProblem
  { odeRhs = odeRhsPure $ \_t x -> coerce (bigA #> (coerce x) + b)
  , odeJacobian = Just (\_t _x -> bigA)
  , odeEvents = mempty
  , odeEventHandler = nilEventHandler
  , odeMaxEvents = 0
  , odeInitCond = bigU0
  , odeSolTimes =vector $ map (deltaT *) [0 .. 10]
  , odeTolerances = defaultTolerances
  }

myOptions :: EncodeOptions
myOptions = defaultEncodeOptions {
      encDelimiter = fromIntegral (ord ' ')
    }

main :: IO ()
main = do
  x <- sol
  writeFile "heat1G.txt" $ encodeWith myOptions $ map toList $ toRows x

defaultOpts :: method -> ODEOpts method
defaultOpts method = ODEOpts
  { maxNumSteps = 1e5
  , minStep     = 1.0e-14
  , fixedStep   = 0
  , maxFail     = 10
  , odeMethod   = method
  , initStep    = Nothing
  , jacobianRepr = SparseJacobian
                 $ SparsePattern
                 $ cmap (fromIntegral :: I -> Int8)
                 $ cmap (\x -> case x of 0 -> 0; _ -> 1)
                 $ flatten
                 $ toInt bigA
  }

instance Element Int8

emptyOdeProblem :: OdeProblem
emptyOdeProblem = OdeProblem
      { odeRhs = error "emptyOdeProblem: no odeRhs provided"
      , odeJacobian = Nothing
      , odeInitCond = error "emptyOdeProblem: no odeInitCond provided"
      , odeEvents = mempty
      , odeTimeBasedEvents = TimeEventSpec $ return $ 1.0 / 0.0
      , odeEventHandler = nilEventHandler
      , odeMaxEvents = 100
      , odeSolTimes = error "emptyOdeProblem: no odeSolTimes provided"
      , odeTolerances = defaultTolerances
      }

nilEventHandler :: EventHandler
nilEventHandler _ _ _ = throwIO $ ErrorCall "nilEventHandler"

defaultTolerances :: Tolerances
defaultTolerances = Tolerances
  { absTolerances = Left 1.0e-6
  , relTolerance = 1.0e-10
  }
