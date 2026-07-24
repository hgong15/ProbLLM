import world
import dataloader
import model
from pprint import pprint

if (
    world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'XING', 'CiteULike', 'LastMF', 'ml-1m', 'amazon-beauty', 'ml-10m', 'ml-10m_1', 'book-crossing']
    or world.dataset.startswith('book-crossing_')
    or world.dataset.startswith('amazon23_')
):
    dataset = dataloader.Loader(path="../data/" + world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}
