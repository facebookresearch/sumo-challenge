#include "sumo/threedee/PointCloud.h"

#include <gtest/gtest.h>

using namespace std;
using namespace sumo;

TEST(PointCloud, Constructor) {
  vector<Vector3> points(3);
  vector<PointCloud::Color> colors(3);
  PointCloud cloud(points, colors);
  ASSERT_EQ(cloud.numPoints(), 3);
  PointCloud::Color expected;
  ASSERT_EQ(expected, cloud.color(0));
}

TEST(PointCloud, Constructor2) {
  vector<Vector3> points(3);
  vector<PointCloud::Color> colors(3);
  PointCloud cloud(points, colors);
  vector<const PointCloud*> clouds = {&cloud, &cloud, &cloud};
  PointCloud tripleCloud(clouds);
  ASSERT_EQ(tripleCloud.numPoints(), 9);
  PointCloud::Color expected;
  ASSERT_EQ(expected, tripleCloud.color(8));
}
