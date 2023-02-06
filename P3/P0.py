# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def is_bw(im):  # Devuelve true si la imagen está en blaco y negro y false en caso contrario
  return len(im.shape) == 2

def bgr2rgb(im): # Dada una imagen con los colores en el modelo BGR, devuelve la misma imagen pero en el modelo RGB
  return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def gray2rgb(im):  # Dada una imagen en blaco y negro, devuelve la misma imagen pero en el modelo RGB
  return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

def readIm(filename, flagColor=1):  # El siguiente método lee una imagen en color si el flagColor es 1 o en blanco y negro si es 0, y la muestra por pantalla
  im = cv2.imread(filename, flagColor)  # Leemos la imagen con el flag correspondiente 0 blanco negro y 1 en color
  if is_bw(im): # Si la imagen está en blaco y negro la mostramos con el mapa de colores "gray"
    plt.imshow(im, cmap = "gray")
  else: # En caso contrario la pasamos al modelo RGB y la mostramos
    im = bgr2rgb(im)
    plt.imshow(im)

  # Las siguientes sentencias sirven para quitar las marcas de los ejes de coordenadas
  plt.xticks([])
  plt.yticks([])

  # Devolvemos la imagen leida
  return im

def rangeDisplay01(im, flag_GLOBAL):  # Normaliza la imagen al rango [0, 1]
  im = im.astype(float) # Pasamos el tipo de la imagen a float
  if is_bw(im) or flag_GLOBAL == 1: # Miramos si la imagen esta en blaco y negro o si el flagGLOBAL es 1
    amax = np.amax(im)  # Conseguimos el máximo de la matriz
    amin = np.amin(im)  # Conseguimos el mínimo de la matriz
    im = (im - amin) / (amax - amin)  # Normalizamos cada dato, dividiendo su diferencia con el mínimo entre la diferencia del máximo y el mínimo
  else: # En caso de que sea una imagen a color, realizaremos esto para cada banda
    for i in range(im.shape[2]):
      amax = np.amax(im[:,:,i])
      amin = np.amin(im[:,:,i])
      im[:,:,i] = (im[:,:,i] - amin) / (amax - amin)
  # Devolvemos la imagen
  return im

def displayIm(im, title='Result',factor= 1, showFlag=True):  
  # Normalize range
  im_out = rangeDisplay01(im, True)
  im_out = im_out[:,:]*255
  # Display the image
  if len(im_out.shape) == 3:
    # im has three channels
    plt.imshow(im_out.astype('uint8')[:,:,::-1])
  else:
    # im has a single channel
    plt.imshow(im_out, cmap='gray')

  figure_size = plt.gcf().get_size_inches()
  plt.gcf().set_size_inches(factor * figure_size)
  plt.title(title) #adding title
  plt.xticks([]), plt.yticks([]) #axis label off
  if showFlag: plt.show()

def pixels(im): # Método auxiliar que devuelve el número de pixeles de una imagen
  return im.shape[0]*im.shape[1]
  
def divisors(integer):  # Método auxiliar que cacula los divisores de un entero
  d = []
  for i in range(1, integer+1): # Para cada entero entre el 1 y el que se pasa como parámetro vemos si ese número es divisor de integer
    if integer%i == 0:
      d.append(i) # En el caso de que lo sea lo añadimos a la lista
  return d

def middle(d):  # Método auxiliar que en el caso de que la lista sea para devuelve los dos elementos del medio y en caso contrario devuelve el elemento del medio dos veces
  if len(d)%2 ==0:
    return [d[len(d)//2-1], d[len(d)//2]]
  else:
    return [d[(len(d)-1)//2], d[(len(d)-1)//2]]

def nr_nc(vim):  # Devuelve el número óptimo de imagenes por fila y columna dado un número de imagenes
  # El método divisors, nos da los divisores del número de imagenes y el método middle nos da los dos divisores para que la forma sea los más cuadrada posible
  d = divisors(len(vim))
  m = middle(d)
  nr = m[0] # Número de imagenes por fila
  nc = m[1] # Número de imagenes por columna
  return nr, nc

def pasteCanvas(canvas): # Método auxiliar que pega una lista de imagenes
  outIm = np.hstack(canvas[0])  # Inicializamos outIm con la primera fila 

  for i in range(1, len(canvas)):  # Contruimos la imagen final
    row = np.hstack(canvas[i])  # Agrupamos todos los elementos de una fila
    outIm = np.vstack([outIm, row]) # Agrupamos la fila con el resto de elementos

  return outIm # Devolvemos el resultado

def displayMI_ES(vim, title='Result',factor= 1):  # Método que muestra varia imagenes con el mismo tamaño y mismo número de bandas
  nr, nc = nr_nc(vim)  # Calcularmos los valores del número de imagenes por fila y columna óptimos (óptimo me refiero que más se asemejan a un cuadrado)

  canvas = [vim[i*nc:(i+1)*nc] for i in range(nr)]  # Definimos el lienzo, creamos listas de tamaño nc con las imagenes tal y como vienen ordenadas en vim

  out = pasteCanvas(canvas)  # Pegamos los trozos con el método auxiliar pasteCanvas()

  return displayIm(out,title,factor)  # Mostramos la imagen

def displayMI_NES(vim): # Método que muestra varia imagenes con distinto tamaño y distinto número de bandas
  for i in range(len(vim)): # Cuando una imagen está en blaco y negro, automaticamente la pasamos a RGB para poder trabajar bien con ella
    if is_bw(vim[i]):
      vim[i] = gray2rgb(vim[i])

  vim.sort(key=pixels, reverse=True)  # Ordenamos el vector de imagenes dependiendo del tamaño en pixeles de cada imagen

  nr, nc = nr_nc(vim) # Calcularmos los valores del número de imagenes por fila y columna óptimos (óptimo me refiero que más se asemejan a un cuadrado)

  # Para cada fila de imagenes vamos a calcular el máximo de las alturas y las vamos a guardar mpr (esto más adelante nos servirá para poder adaptar 
  # las imagenes y que todas tengan la misma altura por fila)
  mpr = np.zeros(nr, dtype=int)
  for i in range(len(vim)):
    mpr[i//nc] = max(mpr[i//nc], vim[i].shape[0])
  maxrow = np.sum(mpr)

  # Para cada columna de imagenes vamos a calcular el máximo de los anchos y los vamos a guardar mpc (esto más adelante nos servirá para poder adaptar 
  # las imagenes y que todas tengan el mismo ancho por columna)
  mpc = np.zeros(nc, dtype=int)
  for i in range(len(vim)):
    mpc[i%nc] = max(mpc[i%nc], vim[i].shape[1])
  maxcolumn = np.sum(mpc)

  # Ajustamos todas las imagenes con la función copyMakeBorder, que creará un borde blaco por debajo y a la derecha de cada imagen que no se ajuste a las medidas máximas en su fila y columna
  wp = np.ones(len(vim[-1].shape))*255  # Pixel blanco que determina el color del borde
  for i in range(len(vim)):
    vim[i] = cv2.copyMakeBorder(vim[i], 0, mpr[i//nc]-vim[i].shape[0], 0, mpc[i%nc]-vim[i].shape[1], cv2.BORDER_CONSTANT, value=wp)

  canvas = [vim[i*nc:(i+1)*nc] for i in range(nr)]  # Definimos el lienzo, creamos listas de tamaño nc con las imagenes tal y como vienen ordenadas en vim

  outIm = pasteCanvas(canvas)  # Pegamos los trozos con el método auxiliar pasteCanvas()

  return outIm[0:maxrow,0:maxcolumn,:]  # Devolvemos la imagen entre los valores de maximo por fila y por columna

def centerSquareCoordinates(im, n): # Método auxiliar que devuelve las coordenadas de un cuadrado centrado de lado n en una imagen
  x = (im.shape[0]+1-n)//2  # Calculamos donde deben comenzar las coordenadas tanto en el eje x como en el y (el +1 es para redondear)
  y = (im.shape[1]+1-n)//2
  xaxis = np.linspace(x, x+n, n, dtype=int) # Definimos los arrays de coordenadas desde el comienzo hasta n posiciones más adelante
  yaxis = np.linspace(y, y+n, n, dtype=int)
  cp = np.array(np.meshgrid(xaxis, yaxis)).T.reshape(-1, 2) # Conseguimos todas las combinaciones posibles para obtener las coordenadas de cada pixel en ese cuadrado
  return cp # Devolvemos el resultado

def pixelArray(p, n): # Método auxilia que devuelve un array de pixeles
  nv = np.array([p for i in range(n)])
  return nv

def changePixelValues(im,cp,nv):  # Función que cambia los pixeles de una imagen por los indicados en un vector de coordenadas y otro de pixeles
  for i in range(len(cp)):  # Recorremos los arrays y cambiamos los valores de cada pixel de la imagen por el nuevo valor
    im[cp[i][0]][cp[i][1]] = nv[i]

  return displayIm(im)  # Mostramos la imagen

def print_images_titles(vim, titles=None, rows=2):  # Función que imprime un grupo de imagenes con sus correspondientes títulos
  fig = plt.gcf() # Con esto obtenemos la figura donde vamos a pinta (figura actual)
  fig.clear() # La limpiamos por si había algo pintado previamente
  columns = int(np.ceil(len(vim)/rows)) # Calculamos el número de imagenes por columnas
  for i in range(len(vim)): # Para cada imagen añadimos un subplot en la posición i+1 con las medidas de imagenes por columna y fila obtenidas
    ax = fig.add_subplot(rows, columns, i+1)
    if is_bw(vim[i]): # Si la imagen está en blanco y negro la mostramos con el mapa de colores "gray"
      ax.imshow(vim[i], cmap='gray')  
    else: # En caso contrario la mostramos tal cual
      ax.imshow(vim[i])
    if titles != None and i < len(titles):  # Si hay título para esta imagen los añadimos
      ax.title.set_text(titles[i])
    ax.set_xticks([]) # Quitamos las marcas de los ejes
    ax.set_yticks([])

  fig.tight_layout()  # Ajustamos el espacio entre imagenes
  plt.show()  # Mostramos la imagen
