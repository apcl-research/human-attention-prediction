import pickle
from metrics_loop import main
import argparse
import csv

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--pids', dest='pids', type=str, default='/nublar/eyeseq/reading/n10/reading_fids.pkl')
    parser.add_argument('--out', dest='out', type=str, default='scanpath_results.csv')
    parser.add_argument('--holdout', dest='holdout', type=str, default='1046788')
    parser.add_argument('--ref-file', dest='ref_dir', type=str, default='/nublar/eyeseq/reading/n10/')
    parser.add_argument('--pred-dir', dest='pred_dir', type=str, default='scanpath_predictions')
    parser.add_argument('--N', dest='N', type=int, default=5)
    
    args = parser.parse_args()
    all_id_filename = args.pids
    out_filename = args.out
    ref_dir = args.ref_dir
    pred_dir = args.pred_dir
    holdout = args.holdout
    N = args.N


    
    all_id = pickle.load(open(all_id_filename,'rb'))

    f = open(out_filename, 'a')

    for pid in all_id:
        reffilename = f'{ref_dir}/byFunction_{pid}/eyeseq_testref.txt'
        predfilename = f'{pred_dir}/predict_{holdout}.txt'
        
        if(pid==holdout):
            normalized_lds, ndcgs, precision_at_ks, recall_at_ks = main(predfilename,  reffilename,'list', N)
            total_normalized_lds = 0
            total_ndcgs = 0
            total_precision_at_ks = 0
            total_recall_at_ks = 0
            for score in normalized_lds:
                total_normalized_lds += score
            for score in ndcgs:
                total_ndcgs += score
            for score in precision_at_ks:
                total_precision_at_ks += score
            for score in recall_at_ks:
                total_recall_at_ks += score


            normalized_ld = total_normalized_lds / len(normalized_lds)
            ndcg = total_ndcgs / len(ndcgs)
            precision_at_k = total_precision_at_ks / len(precision_at_ks)
            recall_at_k = total_recall_at_ks / len(recall_at_ks)

            print(f"Normalized LD:\t{normalized_ld}")
            print(f"NDCG:\t{ndcg}")
            print(f"precision@k:\t{precision_at_k}")
            print(f"recall@k:\t{recall_at_k}")
            new_row = [pid, normalized_ld, ndcg, precision_at_k, recall_at_k]
            with open(out_filename, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(new_row)


        #except:
        #    continue
    f.close()






