from routines import *


RESCALE_SIZE = 256
class Faces(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    def __init__(self, data, cutting=False, blur=False):
        super().__init__()
        # список файлов для загрузки
        self.data = sorted(data)
        self.cutting = cutting
        self.blur = blur



        self.len_ = len(self.data)

    def __len__(self):
        return self.len_


    def load_sample(self, file1):
        image1 = Image.open(file1)
        image1.load()
        return image1.resize((RESCALE_SIZE, RESCALE_SIZE))


    def __getitem__(self, index):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        y = self.load_sample(self.data[index])
        y = transform(y)
        x = create_x(y, self.cutting, self.blur)
        return {'X':x, 'Y':y}
