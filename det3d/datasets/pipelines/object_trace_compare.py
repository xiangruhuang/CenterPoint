import torch

def compute_score(selected_indices):
    return selected_indices.shape[0]

#class ObjectTraceCompare:
#    def __init__(self, num_points, threshold=0.3, device='cpu'):
#        self.num_points = num_points
#        self.visited = torch.zeros(num_points, dtype=torch.long, device=device)
#        self.trace_dict = {}
#        self.threshold = threshold
#
#    def add_object_trace(self, trace_id, trace_dict, conflict_list):
#        trace_dict['score'] = compute_score(trace_dict['point_indices'])
#        self.trace_dict[trace_id] = trace_dict
#        for trace_id in conflict_list:
#            self.trace_dict.pop(trace_id)
#
#    def compare(self, selected_indices):
#        conflict_list = []
#        score = compute_score(selected_indices)
#        for trace_id, trace_dict in self.trace_dict.items():
#            trace_indices = trace_dict['point_indices']
#            self.visited[trace_indices] += 1
#            self.visited[selected_indices] += 1
#            intersection = (self.visited == 2).float().sum()
#            union = (self.visited >= 1).float().sum()
#            iou = (intersection / union).item()
#            self.visited[trace_indices] -= 1
#            self.visited[selected_indices] -= 1
#            if iou > self.threshold:
#                if trace_dict['score'] > score:
#                    return False, []
#                else:
#                    conflict_list.append(trace_id)
#
#        return True, conflict_list

class ObjectTraceCompare:
    def __init__(self, num_points, threshold=0.1, device='cpu'):
        self.num_points = num_points
        self.visited = torch.zeros(num_points, dtype=torch.bool, device=device)
        self.trace_dict = {}
        self.threshold = threshold

    def add_object_trace(self, trace_id, trace_dict, conflict_list):
        trace_dict['score'] = compute_score(trace_dict['point_indices'])
        self.trace_dict[trace_id] = trace_dict
        for trace_id in conflict_list:
            self.trace_dict.pop(trace_id)

    @torch.no_grad()
    def compare(self, selected_indices):
        score = compute_score(selected_indices)
        num_conflict = self.visited[selected_indices].float().sum()
        ratio = num_conflict / selected_indices.shape[0]
        if ratio > self.threshold:
            return False, []
        self.visited[selected_indices] = True
        
        return True, []
