// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SVO_FRAME_HANDLER_H_
#define SVO_FRAME_HANDLER_H_

#include <set>
#include <vikit/abstract_camera.h>
#include <svo/frame_handler_base.h>
#include <svo/reprojector.h>
#include <svo/initialization.h>

namespace svo {

struct Options
{
  /// Maximun duration allowed between keyframes.
  double kfselect_backend_max_time_sec = 3.0;
  
  /// If we are tracking more than this amount of
  /// features, then we don't take a new keyframe.
  size_t kfselect_numkfs_upper_thresh = 110;

  /// If last id - map last id <= this number, then
  /// we don't take a new keyframe.
  int kfselect_min_num_frames_between_kfs = 1; 

  /// If we are tracking less than this amount of
  /// features, then we take a new keyframe.
  size_t kfselect_numkfs_lower_thresh = 80; 

  /// If disparity is less than this number, then we
  /// don't take a new keyframe.
  double kfselect_min_disparity = 30;

  /// If position and orientation is similar to any overlap 
  /// keyframe, then we don't take a new keyframe.
  double kfselect_min_angle = 20;
  double kfselect_min_dist_metric = 0.01;

  /// If update seeds with overlap keyframes when a 
  /// keyframe selected into depth filter.
  bool update_seeds_with_old_keyframes = true;

  /// If update seeds with vogiatizis model, false=Gaussian
  bool use_vogiatzis_update = false;

  /// If change threshold of depth filter every frame.
  bool use_dynamic_thresh = true;

  /// Threshold of depth filter. Greater this number, 
  /// harder converging.
  float seed_convergence_sigma2_thresh = 200.0;
};

/// Monocular Visual Odometry Pipeline as described in the SVO paper.
class FrameHandlerMono : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  FrameHandlerMono(vk::AbstractCamera* cam);
  virtual ~FrameHandlerMono();

  /// Provide an image.
  void addImage(const cv::Mat& img, double timestamp);

  /// Set the first frame (used for synthetic datasets in benchmark node)
  void setFirstFrame(const FramePtr& first_frame);

  /// Get the last frame that has been processed.
  FramePtr lastFrame() { return last_frame_; }

  /// Get the set of spatially closest keyframes of the last frame.
  const set<FramePtr>& coreKeyframes() { return core_kfs_; }

  /// Return the feature track to visualize the KLT tracking during initialization.
  const vector<cv::Point2f>& initFeatureTrackRefPx() const { return klt_homography_init_.px_ref_; }
  const vector<cv::Point2f>& initFeatureTrackCurPx() const { return klt_homography_init_.px_cur_; }

  /// Access the depth filter.
  DepthFilter* depthFilter() const { return depth_filter_; }

  /// An external place recognition module may know where to relocalize.
  bool relocalizeFrameAtPose(
      const int keyframe_id,
      const SE3& T_kf_f,
      const cv::Mat& img,
      const double timestamp);
  
  std::list<int> converged_seed_last_frames_;

protected:
  vk::AbstractCamera* cam_;                     //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
  Reprojector reprojector_;                     //!< Projects points from other keyframes into the current frame
  set<FramePtr> core_kfs_;                      //!< Keyframes in the closer neighbourhood.
  vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?
  initialization::KltHomographyInit klt_homography_init_; //!< Used to estimate pose of the first two keyframes by estimating a homography.
  DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.

  /// Initialize the visual odometry algorithm.
  virtual void initialize();

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first frame until a keyframe is selected.
  virtual UpdateResult processSecondFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();

  /// Try relocalizing the frame at relative position to provided keyframe.
  virtual UpdateResult relocalizeFrame(
      const SE3& T_cur_ref,
      FramePtr ref_keyframe);

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll();

  /// Keyframe selection criterion.
  virtual bool needNewKf(double scene_depth_mean);

  void setCoreKfs(size_t n_closest);

  /// added
  void setThreshold(std::list<int>& converged_seeds,
    float& seed_convergence_sigma2_thresh);

  Options options_;
};

} // namespace svo

#endif // SVO_FRAME_HANDLER_H_
