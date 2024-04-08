# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
import json,uuid,joblib,os,sys
import scipy.spatial as spatial
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import itertools
from datareader import *
from estimater import *
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/mycpp/build')
import yaml



def get_mask(reader, i_frame, ob_id, detect_type):
  if detect_type=='box':
    mask = reader.get_mask(i_frame, ob_id)
    H,W = mask.shape[:2]
    vs,us = np.where(mask>0)
    umin = us.min()
    umax = us.max()
    vmin = vs.min()
    vmax = vs.max()
    valid = np.zeros((H,W), dtype=bool)
    valid[vmin:vmax,umin:umax] = 1
  elif detect_type=='mask':
    mask = reader.get_mask(i_frame, ob_id)
    if mask is None:
      return None
    valid = mask>0
  elif detect_type=='detected':
    mask = cv2.imread(reader.color_files[i_frame].replace('rgb','mask_cosypose'), -1)
    valid = mask==ob_id
  else:
    raise RuntimeError
  return valid



def run_pose_estimation_worker(reader, i_frames, est:FoundationPose=None, debug=0, ob_id=None, device='cuda:0'):
  torch.cuda.set_device(device)
  est.to_device(device)
  est.glctx = dr.RasterizeCudaContext(device=device)

  result = NestDict()
  all_score_result = NestDict()
  all_pose_result = NestDict()
  gt_poses = NestDict()

  for i, i_frame in enumerate(i_frames):
    logging.info(f"{i}/{len(i_frames)}, i_frame:{i_frame}, ob_id:{ob_id}")
    video_id = reader.get_video_id()
    color = reader.get_color(i_frame)
    depth = reader.get_depth(i_frame)
    id_str = reader.id_strs[i_frame]
    H,W = color.shape[:2]

    debug_dir =est.debug_dir

    ob_mask = get_mask(reader, i_frame, ob_id, detect_type=detect_type)
    if ob_mask is None:
      logging.info("ob_mask not found, skip")
      result[video_id][id_str][ob_id] = np.eye(4)
      return result

    est.gt_pose = reader.get_gt_pose(i_frame, ob_id)

    pose,all_scores,all_poses = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=ob_mask, ob_id=ob_id)
    logging.info(f"pose:\n{pose}")

    if debug>=3:
      m = est.mesh_ori.copy()
      tmp = m.copy()
      tmp.apply_transform(pose)
      tmp.export(f'{debug_dir}/model_tf.obj')

    result[video_id][id_str][ob_id] = pose
    all_score_result[video_id][id_str][ob_id] = all_scores
    all_pose_result[video_id][id_str][ob_id] = all_poses
    gt_poses[video_id][id_str][ob_id] = est.gt_pose

  return result, all_scores, all_poses, est.gt_pose
# all_score_result, all_pose_result, gt_poses

def run_pose_estimation():
  wp.force_load(device='cuda')
  reader_tmp = LinemodReader(f'{opt.linemod_dir}/lm_test_all/test/000002', split=None)

  debug = opt.debug
  use_reconstructed_mesh = opt.use_reconstructed_mesh
  debug_dir = opt.debug_dir

  res = NestDict()
  res_scores = NestDict()
  res_poses = NestDict()
  res_gt_poses = NestDict()
  glctx = dr.RasterizeCudaContext()
  mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4),mutable=True)
  mesh_tmp = mesh_tmp.to_mesh()
  est = FoundationPose(model_pts=mesh_tmp.vertices.copy(), model_normals=mesh_tmp.vertex_normals.copy(), symmetry_tfs=None, mesh=mesh_tmp, scorer=None, refiner=None, glctx=glctx, debug_dir=debug_dir, debug=debug)

  final_keep_num = 50
  # reader_tmp.ob_ids
  for ob_id in reader_tmp.ob_ids:
    ob_id = int(ob_id)
    if ob_id==1:
        continue
    if use_reconstructed_mesh:
      mesh = reader_tmp.get_reconstructed_mesh(ob_id, ref_view_dir=opt.ref_view_dir)
    else:
      mesh = reader_tmp.get_gt_mesh(ob_id)
    symmetry_tfs = reader_tmp.symmetry_tfs[ob_id]

    args = []

    video_dir = f'{opt.linemod_dir}/lm_test_all/test/{ob_id:06d}'
    reader = LinemodReader(video_dir, split=None)
    video_id = reader.get_video_id()
    est.reset_object(model_pts=mesh.vertices.copy(), model_normals=mesh.vertex_normals.copy(), symmetry_tfs=symmetry_tfs, mesh=mesh)

    for i in range(len(reader.color_files)):
      args.append((reader, [i], est, debug, ob_id, "cuda:0"))

    outs = []
    out_scores = []
    out_poses = []
    gt_poses = []
    
    
    saved_out_scores = np.empty((len(args),final_keep_num))
    saved_out_poses = np.empty((len(args),final_keep_num,4,4))
    saved_gt_poses = np.empty((len(args),4,4))
    
    for i in range(len(args)):
      out,out_score,out_pose,gt_pose = run_pose_estimation_worker(*args[i])
    #   this_score = out_score[1]['000000'][1]
    #   this_pose = out_pose[1]['000000'][1]
    #   this_gt_pose = gt_pose[1]['000000'][1]
    #   print(this_score)
    #   R = this_pose[:,:3,:3]
    #   t = this_pose[:,:3,3]
    #   R_err = np.linalg.norm(R - this_gt_pose[:3,:3],axis = (1,2))
    #   t_err = np.linalg.norm(t - this_gt_pose[:3,3],axis = 1)
    #   print(R_err)
    #   print(t_err)
      saved_out_scores[i,:] = out_score[:final_keep_num]
      saved_out_poses[i,:,:,:] = out_pose[:final_keep_num,:,:]
      saved_gt_poses[i,:,:] = gt_pose
      
      outs.append(out)
      out_scores.append(out_score)
      out_poses.append(out_pose)
      gt_poses.append(gt_pose)
    
    
    np.save(f'./datas/linemod_out_scores_{ob_id}.npy',saved_out_scores)
    np.save(f'./datas/linemod_out_poses_{ob_id}.npy',saved_out_poses)
    np.save(f'./datas/linemod_gt_poses_{ob_id}.npy',saved_gt_poses)

    # for out in outs:
    #   for video_id in out:
    #     for id_str in out[video_id]:
    #       for ob_id in out[video_id][id_str]:
    #         res[video_id][id_str][ob_id] = out[video_id][id_str][ob_id]

    # for out_score in out_scores:
    #   for video_id in out_score:
    #     for id_str in out_score[video_id]:
    #       for ob_id in out_score[video_id][id_str]:
    #         res_scores[video_id][id_str][ob_id] = out_score[video_id][id_str][ob_id]
            
    # for out_pose in out_poses:
    #   for video_id in out_pose:
    #     for id_str in out_pose[video_id]:
    #       for ob_id in out_pose[video_id][id_str]:
    #         res_poses[video_id][id_str][ob_id] = out_pose[video_id][id_str][ob_id]
            
    # for gt_pose in gt_poses:
    #   for video_id in gt_pose:
    #     for id_str in gt_pose[video_id]:
    #       for ob_id in gt_pose[video_id][id_str]:
    #         res_gt_poses[video_id][id_str][ob_id] = gt_pose[video_id][id_str][ob_id]
  
#   with open('./datas/linemod_res.yml','w') as ff:
#     yaml.safe_dump(make_yaml_dumpable(res), ff)
#   with open('./datas/linemod_res_scores.yml','w') as ff:
#     yaml.safe_dump(make_yaml_dumpable(res_scores), ff)
#   with open('./datas/linemod_res_poses.yml','w') as ff:
#     yaml.safe_dump(make_yaml_dumpable(res_poses), ff)
#   with open('./datas/linemod_res_gt_poses.yml','w') as ff:
#     yaml.safe_dump(make_yaml_dumpable(res_gt_poses), ff)
    # for i in range(len(out_scores)):
    #   for video_id in out_scores[i]:
    #     for id_str in out_scores[i][video_id]:
    #       for ob_id in out_scores[i][video_id][id_str]:
    #         this_scores = out_scores[i][video_id][id_str][ob_id]
    #         this_poses = out_poses[i][video_id][id_str][ob_id]
    #         this_gt_pose = gt_poses[i][video_id][id_str][ob_id]
            
    #         # R_diff = np.dot(this_pose[:3,:3], this_gt_pose[:3,:3].T)
            
    #         # best
    #         indices = np.where(this_scores == this_scores[0])
    #         best_poses = this_poses[indices]
    #         best_R = best_poses[:,:3,:3]
    #         best_t = best_poses[:,:3,3]
            
    #         best_R_err = np.linalg.norm(best_R - this_gt_pose[:3,:3],axis = (1,2))
    #         best_t_err = np.linalg.norm(best_t - this_gt_pose[:3,3],axis = 1)
            
    #         # all
    #         best_R = 0
            
            


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  print(code_dir)
  parser.add_argument('--linemod_dir', type=str, default=f'{code_dir}/dataset', help="linemod root dir")
  parser.add_argument('--use_reconstructed_mesh', type=int, default=0)
  parser.add_argument('--ref_view_dir', type=str, default="/mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16")
  parser.add_argument('--debug', type=int, default=0)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  opt = parser.parse_args()
  set_seed(0)

  detect_type = 'mask'   # mask / box / detected

  run_pose_estimation()
