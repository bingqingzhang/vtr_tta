from config.base_config import Config
from datasets.model_transforms import init_transform_dict
from datasets.msrvtt_dataset import MSRVTTDataset, MSRVTTInferTextDataset, MSRVTTInferVideoDataset
from datasets.msvd_dataset import MSVDDataset, MSVDDatasetInferTextDataset, MSVDDatasetInferVideoDataset
from datasets.lsmdc_dataset import LSMDCDataset, LSMDCInferTextDataset, LSMDCInferVideoDataset
from datasets.activitynet_dataset import ActivityNetDataset, ActivityNetInferTextDataset, ActivityNetInferVideoDataset
from torch.utils.data import DataLoader

class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train', test_time=False):
        img_transforms = init_transform_dict(config.input_res)
        train_img_tfms = img_transforms['clip_train']
        test_img_tfms = img_transforms['clip_test']

        if config.dataset_name == "MSRVTT":
            if split_type == 'train':
                dataset = MSRVTTDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=True, num_workers=config.num_workers)
            else:
                if not test_time:
                    dataset = MSRVTTDataset(config, split_type, test_img_tfms)
                    return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)
                else:
                    video_dataset = MSRVTTInferVideoDataset(config, split_type, test_img_tfms)
                    text_dataset = MSRVTTInferTextDataset(config, split_type)
                    return [DataLoader(video_dataset, batch_size=config.batch_size,
                                       shuffle=False, num_workers=config.num_workers),
                            DataLoader(text_dataset, batch_size=config.batch_size,
                                       shuffle=False, num_workers=config.num_workers)]

        elif config.dataset_name == "MSVD":
            if split_type == 'train':
                dataset = MSVDDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, num_workers=config.num_workers)
            else:
                if not test_time:
                    dataset = MSVDDataset(config, split_type, test_img_tfms)
                    return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)
                else:
                    video_dataset = MSVDDatasetInferVideoDataset(config, split_type, test_img_tfms)
                    text_dataset = MSVDDatasetInferTextDataset(config, split_type)
                    return [DataLoader(video_dataset, batch_size=config.batch_size,
                                       shuffle=False, num_workers=config.num_workers),
                            DataLoader(text_dataset, batch_size=config.batch_size,
                                       shuffle=False, num_workers=config.num_workers)]
                            
        elif config.dataset_name == 'LSMDC':
            if split_type == 'train':
                dataset = LSMDCDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
            else:
                dataset = LSMDCDataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=config.num_workers)
            
        elif config.dataset_name == "ActivityNet":
            if split_type == 'train':
                dataset = ActivityNetDataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
            else:
                if not test_time:
                    dataset = ActivityNetDataset(config, split_type, test_img_tfms)
                    return DataLoader(dataset, batch_size=config.batch_size,
                                shuffle=False, num_workers=config.num_workers)
                else:
                    video_dataset = ActivityNetInferVideoDataset(config, split_type, test_img_tfms)
                    text_dataset = ActivityNetInferTextDataset(config, split_type)
                    return [DataLoader(video_dataset, batch_size=config.batch_size,
                                       shuffle=False, num_workers=config.num_workers),
                            DataLoader(text_dataset, batch_size=config.batch_size,
                                       shuffle=False, num_workers=config.num_workers)]

        else:
            raise NotImplementedError
        
def get_tta_dataset(config, split_type='test'):
    assert split_type == 'test', "TTA is only applicable for test split"
    
    img_transforms = init_transform_dict(config.input_res)
    test_img_tfms = img_transforms['clip_test']
    if config.dataset_name == "MSRVTT":
        video_dataset = MSRVTTInferVideoDataset(config, split_type, test_img_tfms)
        text_dataset = MSRVTTInferTextDataset(config, split_type)
    elif config.dataset_name == "MSVD":
        video_dataset = MSVDDatasetInferVideoDataset(config, split_type, test_img_tfms)
        text_dataset = MSVDDatasetInferTextDataset(config, split_type)
    elif config.dataset_name == "LSMDC":
        video_dataset = LSMDCInferVideoDataset(config, split_type, test_img_tfms)
        text_dataset = LSMDCInferTextDataset(config, split_type)
    elif config.dataset_name == "ActivityNet":
        video_dataset = ActivityNetInferVideoDataset(config, split_type, test_img_tfms)
        text_dataset = ActivityNetInferTextDataset(config, split_type)
    else:
        raise NotImplementedError("TTA is not implemented for this dataset")
    return video_dataset, text_dataset

def get_tta_dataloader(config, query_dataset, gallery_dataset):
    query_dataloader = DataLoader(query_dataset, batch_size=config.batch_size,
                                  num_workers=config.num_workers,
                                  pin_memory=True, drop_last=False)
    gallery_dataloader = DataLoader(gallery_dataset, batch_size=config.batch_size,
                                    num_workers=config.num_workers,
                                    pin_memory=True, drop_last=False)
    return query_dataloader, gallery_dataloader
