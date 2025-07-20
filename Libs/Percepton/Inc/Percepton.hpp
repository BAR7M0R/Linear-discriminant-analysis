/******************************************************************************
 * Copyright 2025, Bartłomiej Głodek
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/


#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP
#include <vector>
#include <functional>
#include <random>
#include <ranges>
#include <numeric>
#include <algorithm>
#include "Matrix.hpp"
#include "Matrix_tools.hpp"


struct Perceptron final
{
    const std::size_t numberOfInputs;
    Matrix<double> v,cv;
    explicit Perceptron(const std::pair<double, double> scopeOfWeightsRandInit = {-1, 1})
    : numberOfInputs(2ull)
    , v(numberOfInputs + 1, 1, 1.0)
    {
        std::random_device rd;
        std::seed_seq seed{rd(),rd(),rd(),rd(),rd(),rd(),rd(),rd()};
        std::mt19937 gen(seed);
        std::uniform_real_distribution<double> dist(scopeOfWeightsRandInit.first,scopeOfWeightsRandInit.second);
        for(std::size_t wi = 0ull; wi< v.rows_; ++wi) v.at(wi) = dist(gen);
        cv = v;
    }
    template<typename Func, typename... Args>
    void train(Func func, Args... args)
    {
        func(this, args...);
    }
};

void perceptronCriterionAlgorithm(Perceptron &perceptron,
                                  const Matrix<double>& inputs,
                                  const std::vector<double>& labels,
                                  const std::size_t maxIter = 100,
                                  const double learningRate = 0.25)
{

    if (inputs.getRowsCount() != perceptron.numberOfInputs) throw std::invalid_argument("wrong number of inputs");
    if (labels.size() != inputs.getColsCount()) throw std::invalid_argument("labels parameter has wrong size");

    Matrix<double> in_temp(1ull, labels.size(), labels);
    for(const auto& r : inputs.getRows())
    {
        in_temp.push_backRow(r);
    }
    in_temp.swapRows(0ull,2ull).swapRows(0ull,1ull);
    for(std::size_t l = 0; l < in_temp.getColsCount(); ++l)
    {
        if (in_temp.at(2ull,l) < 0ull)
        {
            for (std::size_t dr = 0ull; dr < in_temp.getRowsCount()-1ull; ++dr)
            {
                in_temp.at(dr,l)*=-1.0;
            }
        }
    }
    Matrix<bool> Y_filter = (perceptron.v.transpose()*in_temp) < 0.0;
    Matrix Y = (in_temp | static_cast<std::vector<bool>>(Y_filter.getRow(0)));

    for(std::size_t iter = 0; (iter < maxIter) and (Y.getColsCount() != 0); ++iter)
    {
       auto sumof = [&perceptron](const Matrix<double>& in){
           Matrix<double> sum(perceptron.v.getRowsCount(), 1, 0.0);
           for(std::size_t r=0ull ; r< sum.getRowsCount(); ++r)
           {
               for(std::size_t c=0ull ; c < in.getColsCount(); ++c)
               {
                   sum.at(r,0ull)+=in.at(r,c);
               }
           }
           return sum;
       };
       perceptron.v += (sumof(Y) * learningRate);
       Y_filter = (perceptron.v.transpose()*in_temp) < 0.0;
       Y = (in_temp | static_cast<std::vector<bool>>(Y_filter.getRow(0)));
   }

}

void relaxationAlgorithm(Perceptron &perceptron,
                         const Matrix<double>& inputs,
                         const std::vector<double>& labels,
                         const std::size_t maxIter = 100,
                         const double learningRate = 0.25,
                         const double biasValue = 0.5)
{

    if (inputs.getRowsCount() != perceptron.numberOfInputs) throw std::invalid_argument("wrong number of inputs");
    if (labels.size() != inputs.getColsCount()) throw std::invalid_argument("labels parameter has wrong size");

    Matrix<double> in_temp(1ull, labels.size(), labels);
    for(const auto& r : inputs.getRows())
    {
        in_temp.push_backRow(r);
    }
    in_temp.swapRows(0ull,2ull).swapRows(0ull,1ull);
    for(std::size_t l = 0; l < in_temp.getColsCount(); ++l)
    {
        if (in_temp.at(2ull,l) < 0ull)
        {
            for (std::size_t dr = 0ull; dr < in_temp.getRowsCount()-1ull; ++dr)
            {
                in_temp.at(dr,l)*=-1.0;
            }
        }
    }
    Matrix<bool> Y_filter = (perceptron.v.transpose()*in_temp) < biasValue;
    Matrix Y = (in_temp | static_cast<std::vector<bool>>(Y_filter.getRow(0)));
    for(std::size_t iter = 0; (iter < maxIter) and (Y.getColsCount() != 0); ++iter)
    {
        auto sumof = [&perceptron](const Matrix<double>& in, double b){
            Matrix<double> sum(perceptron.v.getRowsCount(), 1, 0.0);
            Matrix in2 = Matrix_tools::sumCols(Matrix_tools::power(in));
            Matrix criteria = perceptron.v.transpose()*in;
            for(std::size_t r=0ull ; r< sum.getRowsCount(); ++r)
            {
                for(std::size_t c=0ull ; c < in.getColsCount(); ++c)
                {
                    sum.at(r, 0ull) += (b-criteria.at(c))/(in2.at(c))*in.at(r,c);
                }
            }
            return sum;
        };
        perceptron.v += (sumof(Y, biasValue) * learningRate);
        Y_filter = (perceptron.v.transpose()*in_temp) < biasValue;
        Y = (in_temp | static_cast<std::vector<bool>>(Y_filter.getRow(0)));
    }
}

#endif //PERCEPTRON_HPP

