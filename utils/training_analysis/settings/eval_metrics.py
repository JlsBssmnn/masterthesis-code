def setting(fig, ax):
    ax.set_title('Training evaluation')
    ax.set_xlabel('total iterations')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=max(1, ax.get_ylim()[1]))
    ax.grid()
