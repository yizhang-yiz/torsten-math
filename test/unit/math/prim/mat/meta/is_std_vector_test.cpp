#include <stan/math/prim/mat.hpp>
#include <gtest/gtest.h>
#include <vector>

TEST(MetaTraits, contains_vector_true) {
  using stan::is_std_vector;
  using std::vector;

  EXPECT_TRUE(is_std_vector<std::vector<double> >::value);
  EXPECT_TRUE(is_std_vector<std::vector<int> >::value);
  EXPECT_TRUE(is_std_vector<std::vector<const double> >::value);
  EXPECT_TRUE(is_std_vector<std::vector<const int> >::value);
  EXPECT_TRUE(is_std_vector<std::vector<Eigen::VectorXd> >::value);
  EXPECT_TRUE(is_std_vector<std::vector<Eigen::RowVectorXd> >::value);
  bool r = is_std_vector<std::vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> > >::value;
  EXPECT_TRUE(r);
}

TEST(MetaTraits, contains_vector_false) {
  using stan::is_std_vector;

  EXPECT_FALSE(is_std_vector<Eigen::VectorXd>::value);
  EXPECT_FALSE(is_std_vector<Eigen::RowVectorXd>::value);
  bool r = is_std_vector<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >::value;
  EXPECT_FALSE(r);
}
