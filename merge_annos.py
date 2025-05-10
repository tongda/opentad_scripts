import os
import json
import glob
from pathlib import Path
import argparse


def merge_annotations(input_dir, output_file):
    """
    合并指定目录下的所有JSON标注文件为一个文件

    Args:
        input_dir (str): 包含JSON标注文件的目录路径
        output_file (str): 输出合并后的JSON文件路径
    """
    # 初始化合并后的数据结构
    merged_data = {
        "database": {}
    }
    
    # 查找目录中的所有JSON文件
    json_files = glob.glob(os.path.join(input_dir, "annotations_*.json"))
    
    if not json_files:
        print(f"警告: 在 {input_dir} 中未找到任何标注文件！")
        return False
    
    print(f"找到 {len(json_files)} 个标注文件")
    
    # 遍历所有JSON文件并合并数据
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            if "database" in data:
                # 合并数据库字段
                for clip_id, clip_info in data["database"].items():
                    if clip_id in merged_data["database"]:
                        print(f"警告: 重复的片段ID: {clip_id}")
                    merged_data["database"][clip_id] = clip_info
            else:
                print(f"警告: 文件 {json_file} 缺少 'database' 字段")
                
            print(f"处理完成: {json_file}")
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
    
    # 保存合并后的数据
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"合并完成! 总共合并了 {len(merged_data['database'])} 个视频片段的标注")
    print(f"合并后的标注文件已保存至: {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='合并视频片段标注文件')
    parser.add_argument('--input_dir', type=str, 
                        default="/mnt/data/dtong/pubrepos/OpenTAD/data/b11_phone_motion2_backview/raw_data/clips2/",
                        help='包含JSON标注文件的目录路径')
    parser.add_argument('--output_file', type=str, 
                        default="/mnt/data/dtong/pubrepos/OpenTAD/data/b11_phone_motion2_backview/annotations/merged_annotations.json",
                        help='输出合并后的JSON文件路径')
    
    args = parser.parse_args()
    
    # 合并标注
    success = merge_annotations(args.input_dir, args.output_file)
    
    # 显示结果统计信息
    if success:
        # 读取合并后的文件进行统计分析
        try:
            with open(args.output_file, 'r') as f:
                merged_data = json.load(f)
                
            total_clips = len(merged_data["database"])
            training_clips = sum(1 for clip_info in merged_data["database"].values() 
                              if clip_info.get("subset") == "training")
            testing_clips = sum(1 for clip_info in merged_data["database"].values() 
                             if clip_info.get("subset") == "testing")
            
            total_annotations = sum(len(clip_info.get("annotations", [])) 
                                  for clip_info in merged_data["database"].values())
            
            print("\n标注统计信息:")
            print(f"总片段数: {total_clips}")
            print(f"训练集片段数: {training_clips}")
            print(f"测试集片段数: {testing_clips}")
            print(f"总标注数: {total_annotations}")
        except Exception as e:
            print(f"生成统计信息时出错: {str(e)}")


if __name__ == "__main__":
    main()