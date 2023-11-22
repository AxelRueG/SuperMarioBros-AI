from utils import ColorMap,DynamicTileType,EnemyType,StaticTileType
import matplotlib.pyplot as plt

def draw_tiles(tiles):

    grid = []

    for row in range(15):
        grid_row = []
        for col in range(16):
            loc = (row, col)
            tile = tiles[loc]

            if isinstance(tile, (StaticTileType, DynamicTileType, EnemyType)):
                rgb = ColorMap[tile.name].value
                grid_row.append(rgb)
            else:
                pass
        grid.append(grid_row)

    plt.figure(1)
    plt.clf()
    plt.imshow(grid, cmap='Greys')
    plt.colorbar()
    plt.pause(0.0000000001)