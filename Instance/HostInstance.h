#pragma once

template<class Implementation>
class HostInstance {
public:
    int size() const {
        return static_cast<Implementation*>(this)->size();
    }
    int edgeWeight(const int from, const int to) const {
        return static_cast<Implementation*>(this)->edgeWeight(from, to);
    }
    int hamiltonianCycleWeight(const int* cycle) const {
        return static_cast<Implementation*>(this)->edgeWeight(cycle);
    }
};
