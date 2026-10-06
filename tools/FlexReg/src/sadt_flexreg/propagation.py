import torch
import vtk
import numpy as np

import functools
import logging

from .errors import ToolUnavailableError

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def require_cuda():
    """Refuse, by name, to propagate a patch without a CUDA device.

    Upstream calls `.cuda()` unconditionally below, and on a CPU-only server
    that surfaced as torch's "Torch not compiled with CUDA enabled" or "No CUDA
    GPUs are available" from deep inside the dilation, read as a bug in the
    tool. Checked once per process: the answer does not change between
    surfaces, and every surface of a batch would otherwise fail the same way.
    """
    if not torch.cuda.is_available():
        raise ToolUnavailableError(
            "the palate patch needs a CUDA GPU and torch sees none on this "
            "server (torch {}, built for CUDA {})".format(
                torch.__version__, torch.version.cuda or "none")
        )


def Difference(t1,t2):
    t1 = t1.unsqueeze(0).expand(len(t2),-1)
    t2 = t2.unsqueeze(1)
    d = torch.count_nonzero(t1 -t2,dim=-1)
    arg = torch.argwhere(d == t1.shape[1])
    dif = torch.unique(t2[arg])
    return dif


def Neighbours(arg_point,F):
    neighbours = torch.tensor([]).cuda()
    F2 = F.unsqueeze(0).expand(len(arg_point),-1,-1)
    arg_point = arg_point.unsqueeze(1).unsqueeze(2)
    arg = torch.argwhere((F2-arg_point) == 0)

    neighbours = torch.unique(F[arg[:,1],:])
    return neighbours


def GetNeighbors(vtkdata, pids_tensor):
    all_neighbor_pids = []

    # Convertir le tensor en une liste d'entiers
    pids_list = pids_tensor.tolist()

    for pid in pids_list:
        cells_id = vtk.vtkIdList()
        vtkdata.GetPointCells(pid, cells_id)

        for ci in range(cells_id.GetNumberOfIds()):
            points_id_inner = vtk.vtkIdList()
            vtkdata.GetCellPoints(cells_id.GetId(ci), points_id_inner)
            for pi in range(points_id_inner.GetNumberOfIds()):
                pid_inner = points_id_inner.GetId(pi)
                if pid_inner != pid:
                    all_neighbor_pids.append(pid_inner)

    # Rendre unique tous les indices de voisins
    unique_neighbors = np.unique(all_neighbor_pids).tolist()
    return torch.tensor(unique_neighbors).cuda().to(torch.int64)




def Dilation(arg_point,F,texture,surf):
    require_cuda()
    arg_point = torch.tensor([arg_point]).cuda().to(torch.int64)
    F = F.cuda()
    texture = texture.cuda()
    neighbour = Neighbours(arg_point,F)
    arg_texture = torch.argwhere(texture == 1).squeeze()
    dif = neighbour.to(torch.int64)
    dif  = Difference(arg_texture,dif)

    dif_queue = [Neighbours(arg_point,F).to(torch.int64)]
    

    nmb_treatment = 1000

    while dif_queue :  # La boucle continue tant que l'une des files d'attente n'est pas vide
        new_neighbour_batch = []
        while dif_queue:
            current_dif = dif_queue.pop(0)
            if current_dif.numel() > nmb_treatment:
                dif_queue.append(current_dif[nmb_treatment:])
                current_dif = current_dif[:nmb_treatment]
            texture[current_dif] = 1
            new_neighbour_batch.append(GetNeighbors(surf,current_dif))
        
        arg_texture = torch.argwhere(texture == 1).squeeze()
        
        while new_neighbour_batch:
            current_neighbours = new_neighbour_batch.pop(0)
            if current_neighbours.numel() > nmb_treatment:
                new_neighbour_batch.append(current_neighbours[nmb_treatment:])
                current_neighbours = current_neighbours[:nmb_treatment]
            dif = Difference(arg_texture, current_neighbours.to(torch.int64))
            if dif.numel() > 0:
                dif_queue.append(dif)
    return texture