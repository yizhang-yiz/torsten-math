#include <stan/math/rev/meta.hpp>
#include <gtest/gtest.h>
#include <test/unit/util.hpp>
#include <vector>

using stan::math::var;
using stan::return_type;

TEST(MetaTraitsRevScal, ReturnTypeVar) {
  test::expect_same_type<var, return_type<var>::type>();
}

TEST(MetaTraitsRevScal, ReturnTypeVarTenParams) {
  test::expect_same_type<var,
                         return_type<double, var, double, int, double, float,
                                     float, float, var, int>::type>();
}


using stan::math::var;
using stan::return_type;
using std::vector;

TEST(MetaTraitsRevArr, ReturnTypeVarArray) {
  test::expect_same_type<var, return_type<vector<var> >::type>();
  test::expect_same_type<var, return_type<vector<var>, double>::type>();
  test::expect_same_type<var, return_type<vector<var>, double>::type>();
}

TEST(MetaTraitsRevArr, ReturnTypeDoubleArray) {
  test::expect_same_type<double, return_type<vector<double> >::type>();
  test::expect_same_type<double, return_type<vector<double>, double>::type>();
  test::expect_same_type<double, return_type<vector<double>, double>::type>();
}
