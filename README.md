# Fixed-wing UAV Sim
A python implementation of a small fixed wing UAV simulator outlined in [Small Unmanned Aircraft: Theory and Practice](https://uavbook.byu.edu/doku.php).

Includes wind simulations, controllers, and a Dubins RRT path planner. 


Run this command for necessary python packages:

```bash
conda create -n path_env python=3.9
conda activate path_env
pip install -r requirements.txt
```

The error: "gl.context == 0":
1. Open OpenGL/context.py
2. Do：
```
# if context == 0:
#     from OpenGL import error
#     raise error.Error(
#         """Attempt to retrieve context when no valid context"""
#     )
```