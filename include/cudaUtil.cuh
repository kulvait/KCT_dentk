#pragma once

inline dim3 getNumBlocks(dim3 threads, int SIZEX, int SIZEY)
{
    return dim3((SIZEY + threads.x - 1) / threads.x,
                (SIZEX + threads.y - 1) / threads.y);
}
