import cv2 as cv
import glob
import matplotlib.animation as animation
from joblib import Parallel, delayed
from PIL import Image
from matplotlib import pyplot as plt


def load_images() -> list:
    images = []
    for i in range(1, 323):
        images.append(cv.imread(f'quadros_cinza/frame_{i:04}.png', cv.IMREAD_GRAYSCALE))
    
    images_copies = []
    for img in images:
        images_copies.append(img.copy())
        
    return images, images_copies

def load_templates() -> list:
    templates = []
    labels = ["0054", "0170", "0275"]
    for label in labels:
        template = cv.imread(f'templates_cinza/template_{label}.png', cv.IMREAD_GRAYSCALE)
        templates.append(template)
    
    ws_hs = []
    for t in templates:
        w, h = t.shape[::-1]
        ws_hs.append((w, h))
    
    return templates, ws_hs

def enumerate_images_with_templates(templates, templates_batches):
    enumerated_images = []
    
    for i in range(len(templates)):    
        for j in range(templates_batches[i][0], templates_batches[i][1]):
            enumerated_images.append((i, j))
    
    return enumerated_images

def main():
    images, images_copies = load_images()

    templates, ws_hs = load_templates()

    # Metodos que serão testados
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    # Cria uma lista para cada metodo, onde cada posicao conterá uma dupla (min, max)
    methods_tables = dict(zip(methods, [[] for _ in range(len(methods))]))

    # Marca onde inicia e termina os frames de cada carro.
    # Como não se achou uma maneira melhor de fazer isso, elas foram marcadas
    # manualmente.
    templates_batches = [(0, 83), (84, 189), (190, 322)]

    # Associa os templates aos seus respectivos batches
    enumerated_images = enumerate_images_with_templates(templates, templates_batches)

    # Aplica cada método sobre a detecção de cada frame
    # Ao final de um método, obtemos a sua tabela preenchida
    # e as imagens com o retângulo demonstrando onde o 
    # template match ficou
    for meth in methods:
        for templ_ind, img_ind in enumerated_images:
            img = images_copies[img_ind].copy()
            method = eval(meth)
            
            res = cv.matchTemplate(img, templates[templ_ind], method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            
            methods_tables[meth].append((min_val, max_val))
            
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
                bottom_right = (top_left[0] + ws_hs[templ_ind][0], top_left[1] + ws_hs[templ_ind][1])
            
            cv.rectangle(img,top_left, bottom_right, 255, 2)
            
            fig, ax = plt.subplots()

            ax.imshow(img, cmap='gray')

            ax.axis('off')

            fig.savefig(f'rastreio/{meth}/rast_{img_ind:04}.png', bbox_inches='tight', pad_inches=0)

            plt.close(fig)

    # Gera os gráficos de cada método
    for meth in methods:
        min_values = [t[0] for t in methods_tables[meth]]
        max_values = [t[1] for t in methods_tables[meth]]
        
        x_values = list(range(len(methods_tables[meth])))

        plt.figure(figsize=(10, 5))
        plt.plot(x_values, min_values, label='Min Values', marker='o', linestyle='-', color='b')
        plt.plot(x_values, max_values, label='Max Values', marker='o', linestyle='-', color='r')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(meth)
        plt.legend()
        plt.show()

    #############################################################

    # Gera as imagens de cada método
    for meth in methods:
        images = [Image.open(img) for img in sorted(glob.glob(f'rastreio/{meth}/rast_*.png'))]

        fig = plt.figure()

        ims = []
        for img in images:
            im = plt.imshow(img, animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat_delay=1000)

        plt.show()

        ani.save(f"animations/animation_{meth}.gif")


if __name__ == '__main__':
    main()
