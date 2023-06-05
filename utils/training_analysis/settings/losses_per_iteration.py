def setting(fig, ax):
    ax.set_title('Training losses')
    ax.set_xlabel('total iterations')
    ax.set_ylabel('loss')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=min(ax.get_ylim()[1], 3))
    ax.grid()
