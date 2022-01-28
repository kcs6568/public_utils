import os
import json
import logging
import torch.distributed as dist

def get_root_logger(log_dir, case=None):
    handlers = []
    
    logger = logging.getLogger()

    # 로그의 출력 기준 설정
    logger.setLevel(logging.INFO)

    # log 출력 형식
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # log 출력
    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    log_path = log_dir
    if (case is not None) and (not log_dir.endswith('log')):
        log_path = os.path.join(log_dir, f'{case}.log')        
    if rank == 0:
        file_handler = logging.FileHandler(log_path)
        handlers.append(file_handler)
        
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)
    
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)

    # # log를 파일에 출력
    
    # file_handler = logging.FileHandler(log_path)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    
    return logger

from torch.utils.tensorboard import SummaryWriter
class TensorBoardLogger(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix=''):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        # if dist.is_available() and dist.is_initialized():
        #     rank = dist.get_rank()
        # else:
        #     rank = 0
        
    def write_train_value(self, val_dict):
        self.add_scalar("Loss/train", val_dict['loss'], val_dict['epoch'])
        self.add_scalar("Pred@1/train", val_dict['t1'], val_dict['epoch'])
        self.add_scalar("Pred@5/train", val_dict['t5'], val_dict['epoch'])
        
        
    def write_test_value(self, val_dict):
        self.add_scalar("Loss/test", val_dict['loss'], val_dict['epoch'])
        self.add_scalar("Pred@1/test", val_dict['t1'], val_dict['epoch'])
        self.add_scalar("Pred@5/test", val_dict['t5'], val_dict['epoch'])
        
        
    def close_tb(self):
        self.close()
        
        
class ResultStorage():
    def __init__(self, filepath) -> None:
        self.filepath = filepath
        self.final_results = dict(results=[])
        self.init_setup()
        
    
    def init_setup(self):
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []
        self.test_acc = []
        self.test_loss = []
        
        self.top1 = 0.
        self.top5 = 0.
        self.loss = 0.
        
        
    def set_train_value(self, val_dict):
        if not self._check_keys(val_dict):
            raise RuntimeError("the keys is not matched")
        
        self.top1 = val_dict['t1']
        self.top5 = val_dict['t5']
        self.loss = val_dict['loss']
        
        if not self._check_datatype():
            raise TypeError("the result data type is not float.")
        
        self.train_acc.append([self.top1, self.top5])
        self.train_loss.append(self.loss)
        
        # self.dump(dict(train_acc=self.train_acc))
        # self.dump(dict(train_loss=self.train_loss))

    
    def set_test_value(self, val_dict):
        if not self._check_keys(val_dict):
            raise RuntimeError("the keys is not matched")
        
        self.top1 = val_dict['t1']
        self.top5 = val_dict['t5']
        self.loss = val_dict['loss']
        
        if not self._check_datatype():
            raise TypeError("the result data type is not float.")
        
        self.test_acc.append([self.top1, self.top5])
        self.test_loss.append(self.loss)
        
        # self.dump(dict(test_acc=self.test_acc))
        # self.dump(dict(test_loss=self.test_loss)) 
        
        
    def set_val_value(self, val_dict):
        if not self._check_keys(val_dict):
            raise RuntimeError("the keys is not matched")
        
        self.top1 = val_dict['t1']
        self.top5 = val_dict['t5']
        self.loss = val_dict['loss']
        
        if not self._check_datatype():
            raise TypeError("the result data type is not float.")
        
        self.val_acc.append([self.top1, self.top5])
        self.val_loss.append(self.loss)
        
        # self.dump(dict(val_acc=self.val_acc))
        # self.dump(dict(val_loss=self.val_loss))    

        
    def dump(self, value):
        with open(self.filepath, 'w') as f:
            json.dump(value, f, indent=2)
            
    
    def _check_keys(self, val_dict):
        val_keys = list(val_dict.keys())
        assert len(self.keys) == len(val_keys)
        return sorted(self.keys) == sorted(val_keys)      
    
    
    def _check_datatype(self):
        if (
            isinstance(self.top1, float) and
            isinstance(self.top5, float) and
            isinstance(self.loss, float)):
            return True
        else:
            return False
        
          
    

class ResultWriter(ResultStorage):
    def __init__(self, filepath, keys) -> None:
        super().__init__(filepath)
        self.keys = keys
        self.best_prec1 = 0.
        self.best_prec5 = 0.
    
    def write_epoch_result(self, epoch):
        self.final_results['results'].append(
            dict(epoch=epoch,
                 result=self._merge_results(epoch))
        )
        
        self.dump(self.final_results)
    
        
    def _merge_results(self, epoch):
        return dict(
            # train_acc=self.train_acc,
            # train_loss=self.train_loss,
            # test_acc=self.test_acc,
            # test_loss=self.test_loss,
            # val_acc=self.val_acc,
            # val_loss=self.val_loss,
            # best_t1=self.best_prec1,
            # best_t5=self.best_prec5
            train_acc=self.train_acc[epoch],
            train_loss=self.train_loss[epoch],
            val_acc=self.val_acc[epoch],
            val_loss=self.val_loss[epoch],
            best_t1=self.best_prec1,
            best_t5=self.best_prec5
        )
        
            
    
    
        
        
    
    
# class TrainWriter(ResultStorage):
#     def __init__(self, filepath) -> None:
#         super().__init__(filepath)
        
        
#     def write_result(self, val):
#         self.write_train_value(val)
        
        
        
# class ValWriter(ResultStorage):
#     def __init__(self, filepath) -> None:
#         super().__init__(filepath)
        
        
#     def write_result(self, val):
#         self.write_val_value(val)        
        
        
# class TestWriter(ResultStorage):
#     def __init__(self, filepath) -> None:
#         super().__init__(filepath)
        
        
#     def write_result(self, val):
#         self.write_test_value(val)
    
        
